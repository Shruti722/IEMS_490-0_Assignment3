import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    split: str = "train"
    text_field_chosen: str = "chosen"
    text_field_rejected: str = "rejected"
    prompt_field: str = "prompt"
    model_name: str = "gpt2"
    max_length: int = 512
    val_ratio: float = 0.1
    seed: int = 42


def load_hh_dataset(cfg: DataConfig):
    ds = load_dataset(cfg.dataset_name, split=cfg.split)
    ds = ds.shuffle(seed=cfg.seed)
    return ds


def build_tokenizer(cfg: DataConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def concat_prompt_answer(prompt: str, answer: str) -> str:
    if prompt is None:
        return answer
    return prompt + "\n\nAssistant: " + answer


def preprocess_pairs(ds, cfg: DataConfig, tokenizer) -> Tuple[Dict, Dict]:
    """
    Returns tokenized train and val dicts:
    {
      "input_ids_chosen": ...,
      "input_ids_rejected": ...,
      "attention_mask_chosen": ...,
      "attention_mask_rejected": ...
    }
    """

    def _tokenize(example):
        prompt = example.get(cfg.prompt_field, "")
        chosen = example[cfg.text_field_chosen]
        rejected = example[cfg.text_field_rejected]

        text_chosen = concat_prompt_answer(prompt, chosen)
        text_rejected = concat_prompt_answer(prompt, rejected)

        tok_chosen = tokenizer(
            text_chosen,
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        tok_rejected = tokenizer(
            text_rejected,
            max_length=cfg.max_length,
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
    n_total = len(tokenized)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val

    train_ds = tokenized.select(range(n_train))
    val_ds = tokenized.select(range(n_train, n_total))

    return train_ds, val_ds


def build_prompt_only_dataset(ds, cfg: DataConfig, tokenizer):
    """
    Used for supervised fine tuning and RL generation.
    Returns prompts and chosen answers tokenized.
    """

    def _process(example):
        prompt = example.get(cfg.prompt_field, "")
        chosen = example[cfg.text_field_chosen]

        prompt_ids = tokenizer(
            prompt,
            max_length=cfg.max_length // 2,
            truncation=True,
            padding="max_length",
        )
        answer_ids = tokenizer(
            chosen,
            max_length=cfg.max_length // 2,
            truncation=True,
            padding="max_length",
        )

        return {
            "prompt_ids": prompt_ids["input_ids"],
            "prompt_mask": prompt_ids["attention_mask"],
            "answer_ids": answer_ids["input_ids"],
            "answer_mask": answer_ids["attention_mask"],
        }

    proc = ds.map(_process, batched=False)
    return proc
