from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data_utils import (
    DataConfig,
    load_hh_dataset,
    build_tokenizer,
    build_prompt_only_dataset,
)
from reward_model import RewardModel, RewardConfig


# =========================
# Supervised Fine-Tuning (SFT)
# =========================

@dataclass
class SFTConfig:
    model_name: str = "gpt2"
    batch_size: int = 2
    lr: float = 1e-5
    num_epochs: int = 1
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def supervised_finetune():
    data_cfg = DataConfig()
    ds = load_hh_dataset(data_cfg)
    ds = ds.select(range(500))  # subset for speed

    tokenizer = build_tokenizer(data_cfg)
    proc = build_prompt_only_dataset(ds, data_cfg, tokenizer)

    sft_cfg = SFTConfig(model_name=data_cfg.model_name)
    device = sft_cfg.device

    model = AutoModelForCausalLM.from_pretrained(sft_cfg.model_name).to(device)

    def collate_batch(batch):
        input_ids = []
        attn_masks = []
        labels = []
        for ex in batch:
            prompt_ids = ex["prompt_ids"]
            answer_ids = ex["answer_ids"]
            seq = prompt_ids + answer_ids
            seq = seq[: sft_cfg.max_length]
            mask = [1] * len(seq)
            labels_seq = [-100] * len(prompt_ids) + answer_ids
            labels_seq = labels_seq[: sft_cfg.max_length]

            input_ids.append(seq)
            attn_masks.append(mask)
            labels.append(labels_seq)

        max_len = max(len(x) for x in input_ids)

        def pad(seq, pad_val):
            return seq + [pad_val] * (max_len - len(seq))

        input_ids = torch.tensor([pad(x, tokenizer.pad_token_id) for x in input_ids])
        attn_masks = torch.tensor([pad(x, 0) for x in attn_masks])
        labels = torch.tensor([pad(x, -100) for x in labels])

        return {"input_ids": input_ids, "attention_mask": attn_masks, "labels": labels}

    loader = DataLoader(
        proc, batch_size=sft_cfg.batch_size, shuffle=True, collate_fn=collate_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.lr)
    num_training_steps = sft_cfg.num_epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    model.train()
    step = 0
    for epoch in range(sft_cfg.num_epochs):
        for batch in loader:
            step += 1
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"[SFT] step {step} loss {loss.item():.4f}")

    save_path = "../models/policy_sft"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved SFT policy to {save_path}")


# =========================
# Helper: sampling and logprobs (no .generate)
# =========================

@dataclass
class PPOConfig:
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    lr: float = 1e-5
    num_updates: int = 10   # shorter for Colab
    max_gen_len: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def sample_sequences(model, tokenizer, prompts: List[str], max_len: int, device: str):
    """
    Sample sequences token by token (no grad) and return:
    - responses: decoded strings
    - ids_list: list of 1D LongTensor token id sequences
    """
    model.eval()
    responses = []
    ids_list = []

    for p in prompts:
        if not isinstance(p, str) or len(p.strip()) == 0:
            prompt_text = "Human: Hello\nAssistant:"
        else:
            prompt_text = p

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        with torch.no_grad():
            for _ in range(max_len):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits[:, -1, :]  # last token logits
                logp = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(logp)
                next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attn_mask = torch.cat(
                    [attn_mask, torch.ones_like(next_token, device=device)], dim=-1
                )

        full_ids = input_ids[0].detach().cpu()
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        responses.append(text)
        ids_list.append(full_ids)

    return responses, ids_list


def compute_seq_logprob(model, ids_list: List[torch.Tensor], device: str):
    """
    Given a list of token-id sequences, compute mean logprob per sequence
    under the given model (with grad).
    """
    model.eval()
    logps = []
    for ids in ids_list:
        ids = ids.unsqueeze(0).to(device)  # (1, L)
        attn = torch.ones_like(ids).to(device)
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        labels = ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        seq_logp = token_logp.mean(dim=-1)  # (1,)
        logps.append(seq_logp.squeeze(0))
    return torch.stack(logps)  # (batch,)


# =========================
# PPO training
# =========================

def ppo_train():
    data_cfg = DataConfig()
    ds = load_hh_dataset(data_cfg)
    ds = ds.select(range(200))  # subset for speed

    device = PPOConfig().device

    policy = AutoModelForCausalLM.from_pretrained("../models/policy_sft").to(device)
    ref_policy = AutoModelForCausalLM.from_pretrained("../models/policy_sft").to(device)
    ref_policy.eval()

    tokenizer = AutoTokenizer.from_pretrained("../models/policy_sft")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_cfg = RewardConfig()
    reward_model = RewardModel(reward_cfg.model_name)
    reward_state = torch.load("../models/reward_model/reward_head.bin", map_location="cpu")
    reward_model.load_state_dict(reward_state, strict=False)
    reward_model.backbone = AutoModelForCausalLM.from_pretrained("../models/reward_model")
    reward_model.to(device)
    reward_model.eval()

    ppo_cfg = PPOConfig()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo_cfg.lr)

    prompts = [ex.get(data_cfg.prompt_field, "") for ex in ds]

    for update in range(ppo_cfg.num_updates):
        prompt = [prompts[update % len(prompts)]]

        # 1) Sample sequences (no grad)
        responses, ids_list = sample_sequences(
            policy, tokenizer, prompt, ppo_cfg.max_gen_len, device=device
        )

        # 2) logprob under old policy (detached)
        logp_old = compute_seq_logprob(policy, ids_list, device=device).detach()

        # 3) reward from reward model (no grad)
        with torch.no_grad():
            full_text = responses[0]
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=data_cfg.max_length,
            ).to(device)
            r = reward_model(inputs["input_ids"], inputs["attention_mask"])
        reward = r.detach()

        advantage = reward  # simple baseline

        # 4) logprob under current policy (with grad)
        logp_new = compute_seq_logprob(policy, ids_list, device=device)

        # 5) logprob under ref policy (no grad, for KL)
        with torch.no_grad():
            logp_ref = compute_seq_logprob(ref_policy, ids_list, device=device)

        ratio = torch.exp(logp_new - logp_old)
        unclipped = ratio * advantage
        clipped = torch.clamp(
            ratio, 1 - ppo_cfg.clip_epsilon, 1 + ppo_cfg.clip_epsilon
        ) * advantage
        ppo_loss = -torch.min(unclipped, clipped).mean()

        kl = (logp_new - logp_ref).mean()
        kl_loss = ppo_cfg.kl_coef * kl

        loss = ppo_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (update + 1) % 2 == 0:
            print(
                f"[PPO] update {update + 1} loss {loss.item():.4f} "
                f"reward {reward.item():.4f} kl {kl.item():.4f}"
            )

    save_path = "../models/policy_ppo"
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved PPO policy to {save_path}")


# =========================
# GRPO training
# =========================

@dataclass
class GRPOConfig:
    group_size: int = 4
    lr: float = 1e-5
    num_updates: int = 10
    max_gen_len: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def grpo_train():
    data_cfg = DataConfig()
    ds = load_hh_dataset(data_cfg)
    ds = ds.select(range(200))

    device = GRPOConfig().device

    policy = AutoModelForCausalLM.from_pretrained("../models/policy_sft").to(device)
    tokenizer = AutoTokenizer.from_pretrained("../models/policy_sft")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_cfg = RewardConfig()
    reward_model = RewardModel(reward_cfg.model_name)
    reward_state = torch.load("../models/reward_model/reward_head.bin", map_location="cpu")
    reward_model.load_state_dict(reward_state, strict=False)
    reward_model.backbone = AutoModelForCausalLM.from_pretrained("../models/reward_model")
    reward_model.to(device)
    reward_model.eval()

    grpo_cfg = GRPOConfig()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=grpo_cfg.lr)

    prompts = [ex.get(data_cfg.prompt_field, "") for ex in ds]

    for update in range(grpo_cfg.num_updates):
        prompt = [prompts[update % len(prompts)]]

        responses_list = []
        logprobs_list = []
        rewards_list = []

        # group of samples for same prompt
        for _ in range(grpo_cfg.group_size):
            responses, ids_list = sample_sequences(
                policy, tokenizer, prompt, grpo_cfg.max_gen_len, device=device
            )
            responses_list.append(responses[0])
            # compute logprob with grad
            logp = compute_seq_logprob(policy, ids_list, device=device)
            logprobs_list.append(logp[0].unsqueeze(0))

        logprobs = torch.cat(logprobs_list, dim=0)

        with torch.no_grad():
            for full_text in responses_list:
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=data_cfg.max_length,
                ).to(device)
                r = reward_model(inputs["input_ids"], inputs["attention_mask"])
                rewards_list.append(r[0])

        rewards = torch.stack(rewards_list)
        group_mean = rewards.mean()
        advantages = rewards - group_mean

        loss = -(logprobs * advantages.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (update + 1) % 2 == 0:
            print(
                f"[GRPO] update {update + 1} loss {loss.item():.4f} "
                f"group_mean_reward {group_mean.item():.4f}"
            )

    save_path = "../models/policy_grpo"
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved GRPO policy to {save_path}")


# =========================
# DPO training
# =========================

@dataclass
class DPOConfig:
    model_name: str = "gpt2"
    batch_size: int = 2
    lr: float = 1e-5
    num_epochs: int = 1
    beta: float = 0.1
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def dpo_train():
    data_cfg = DataConfig()
    ds = load_hh_dataset(data_cfg)
    ds = ds.select(range(500))

    tokenizer = build_tokenizer(data_cfg)

    def _tokenize(example):
        prompt = example.get(data_cfg.prompt_field, "")
        chosen = example[data_cfg.text_field_chosen]
        rejected = example[data_cfg.text_field_rejected]

        text_chosen = prompt + "\n\nAssistant: " + chosen
        text_rejected = prompt + "\n\nAssistant: " + rejected

        tok_chosen = tokenizer(
            text_chosen,
            max_length=data_cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        tok_rejected = tokenizer(
            text_rejected,
            max_length=data_cfg.max_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids_chosen": tok_chosen["input_ids"],
            "attention_mask_chosen": tok_chosen["attention_mask"],
            "input_ids_rejected": tok_rejected["input_ids"],
            "attention_mask_rejected": tok_rejected["attention_mask"],
        }

    tokenized = ds.map(_tokenize, batched=False)

    # make dataset return tensors so .to(device) works
    cols = [
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
    ]
    tokenized.set_format(type="torch", columns=cols)

    dpo_cfg = DPOConfig(model_name=data_cfg.model_name)
    device = dpo_cfg.device

    policy = AutoModelForCausalLM.from_pretrained("../models/policy_sft").to(device)
    ref_policy = AutoModelForCausalLM.from_pretrained("../models/policy_sft").to(device)
    ref_policy.eval()

    loader = DataLoader(tokenized, batch_size=dpo_cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=dpo_cfg.lr)
    num_training_steps = dpo_cfg.num_epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    def seq_logprob(model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        attn = attention_mask[:, 1:]
        token_logp = token_logp * attn
        seq_logp = token_logp.sum(dim=-1) / (attn.sum(dim=-1) + 1e-8)
        return seq_logp

    policy.train()
    step = 0
    for epoch in range(dpo_cfg.num_epochs):
        for batch in loader:
            step += 1
            optimizer.zero_grad()
            ic = batch["input_ids_chosen"].to(device)
            ac = batch["attention_mask_chosen"].to(device)
            ir = batch["input_ids_rejected"].to(device)
            ar = batch["attention_mask_rejected"].to(device)

            log_pi_c = seq_logprob(policy, ic, ac)
            log_pi_r = seq_logprob(policy, ir, ar)

            with torch.no_grad():
                log_ref_c = seq_logprob(ref_policy, ic, ac)
                log_ref_r = seq_logprob(ref_policy, ir, ar)

            beta = dpo_cfg.beta
            diff_pi = log_pi_c - log_pi_r
            diff_ref = log_ref_c - log_ref_r
            diff = beta * (diff_pi - diff_ref)
            loss = -torch.nn.functional.logsigmoid(diff).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"[DPO] step {step} loss {loss.item():.4f}")

    save_path = "../models/policy_dpo"
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved DPO policy to {save_path}")
