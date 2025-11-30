from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from data_utils import DataConfig, load_hh_dataset, build_tokenizer, preprocess_pairs


@dataclass
class RewardConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 4
    lr: float = 1e-5
    num_epochs: int = 1
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel(nn.Module):
    """
    Reward model = GPT-2 backbone + linear value head
    Output: scalar reward for each input sequence.
    """
    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.n_embd
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Use backbone's transformer to get hidden states
        outputs = self.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state        # (batch, seq, hidden)
        last_token = last_hidden[:, -1, :]             # (batch, hidden)
        reward = self.value_head(last_token).squeeze(-1)  # (batch,)
        return reward


def compute_pairwise_loss(reward_chosen, reward_rejected):
    """
    L = -log( sigmoid(r_chosen - r_rejected) )
    """
    diff = reward_chosen - reward_rejected
    return -torch.nn.functional.logsigmoid(diff).mean()


def compute_accuracy(reward_chosen, reward_rejected):
    return (reward_chosen > reward_rejected).float().mean().item()


def train_reward_model():
    # 1. Load & preprocess data
    data_cfg = DataConfig()
    ds = load_hh_dataset(data_cfg)
    # use a subset for speed (you can increase this if Colab can handle it)
    ds = ds.select(range(1000))

    tokenizer = build_tokenizer(data_cfg)
    train_ds, val_ds = preprocess_pairs(ds, data_cfg, tokenizer)

    # >>> THIS PART IS NEW: make dataset return torch tensors <<<
    cols = [
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
    ]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    # <<< END NEW PART >>>

    # 2. Model & training setup
    reward_cfg = RewardConfig(model_name=data_cfg.model_name)
    device = reward_cfg.device

    model = RewardModel(reward_cfg.model_name).to(device)

    train_loader = DataLoader(train_ds, batch_size=reward_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=reward_cfg.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=reward_cfg.lr, weight_decay=reward_cfg.weight_decay)
    num_training_steps = reward_cfg.num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    # 3. Train
    global_step = 0
    model.train()
    for epoch in range(reward_cfg.num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids_c = batch["input_ids_chosen"].to(device)
            mask_c = batch["attention_mask_chosen"].to(device)
            input_ids_r = batch["input_ids_rejected"].to(device)
            mask_r = batch["attention_mask_rejected"].to(device)

            r_c = model(input_ids_c, mask_c)
            r_r = model(input_ids_r, mask_r)
            loss = compute_pairwise_loss(r_c, r_r)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                acc = compute_accuracy(r_c.detach(), r_r.detach())
                print(f"[Reward] step {global_step} loss {loss.item():.4f} acc {acc:.4f}")

    # 4. Validation
    model.eval()
    all_acc = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids_c = batch["input_ids_chosen"].to(device)
            mask_c = batch["attention_mask_chosen"].to(device)
            input_ids_r = batch["input_ids_rejected"].to(device)
            mask_r = batch["attention_mask_rejected"].to(device)

            r_c = model(input_ids_c, mask_c)
            r_r = model(input_ids_r, mask_r)
            acc = compute_accuracy(r_c, r_r)
            all_acc.append(acc)
    val_acc = sum(all_acc) / len(all_acc)
    print(f"[Reward] Validation accuracy: {val_acc:.4f}")

    # 5. Save
    save_path = "../models/reward_model"
    model.backbone.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), f"{save_path}/reward_head.bin")
    print(f"Saved reward model to {save_path}")


if __name__ == "__main__":
    train_reward_model()
