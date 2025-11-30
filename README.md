# IEMS_490-0_Assignment3

This repository contains my implementation for **LLM Assignment 3**, where I:

1. Train a **reward model** on the Anthropic HH-RLHF dataset  
2. Train a **supervised fine-tuned policy (SFT)** starting from GPT-2  
3. Train three RLHF-style policies on top of SFT:
   - **PPO**
   - **GRPO**
   - **DPO**
4. Evaluate all models quantitatively (reward, KL) and qualitatively (generated samples and error analysis)

All experiments were run on **Google Colab with a T4 GPU**, but the code can also be run locally with a GPU.

---

## Repository structure

```text
LLM_Assignment_3/
├── ANLYSIS.md                   # Write-up for Part 4 (quantitative + qualitative analysis)
├── LLM_Assignment_3.ipynb       # Colab notebook used to run the full pipeline
├── README.md                    # This file
├── analysis/
│   └── reward_model_mistakes.jsonl   # 20+ error cases from the reward model
├── data/                        # (Optional) place for any cached datasets
├── models/
│   ├── policy_dpo/              # DPO-trained policy (HF format)
│   ├── policy_grpo/             # GRPO-trained policy (HF format)
│   ├── policy_ppo/              # PPO-trained policy (HF format)
│   ├── policy_sft/              # Supervised fine-tuned policy (HF format)
│   └── reward_model/            # Reward model (HF format + reward_head.bin)
├── samples/
│   ├── base_gpt2_samples.jsonl  # ~20 generations from base GPT-2
│   ├── policy_dpo_samples.jsonl # ~20 generations from DPO policy
│   ├── policy_grpo_samples.jsonl# ~20 generations from GRPO policy
│   ├── policy_ppo_samples.jsonl # ~20 generations from PPO policy
│   └── policy_sft_samples.jsonl # ~20 generations from SFT policy
└── src/
    ├── data_utils.py            # Dataset loading & preprocessing utilities
    ├── reward_model.py          # Reward model definition & training code
    ├── rl_algorithms.py         # SFT, PPO, GRPO, and DPO training code
    └── train_all.py             # (Optional) script that runs all training stages in sequence
```

> **Note:** The trained models in `models/` are in standard Hugging Face format (config, tokenizer, model.safetensors, etc.), plus an extra `reward_head.bin` for the reward model.

---

## Environment and dependencies

The code assumes:

- **Python:** 3.10+  
- **GPU:** CUDA-capable device (e.g., NVIDIA T4 on Colab)  
- **Core Python libraries:**
  - `torch`
  - `transformers`
  - `datasets`
  - `accelerate`
  - `numpy` (transitive)

### Installing dependencies (local)

Create and activate a virtual environment (optional but recommended), then:

```bash
pip install torch transformers datasets accelerate
```

On Colab, most dependencies are already available; you may only need:

```python
!pip install -q transformers datasets accelerate
```

---

## How to run the assignment

You can either:

1. **Use the notebook** `LLM_Assignment_3.ipynb` (recommended for Colab), or  
2. **Run the Python modules directly** from the `src/` directory.

### 1. Running everything in Google Colab

1. Upload the project folder to your Google Drive (or clone the repo in Colab).
2. Mount Drive and set the working directory inside the notebook:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   ASSIGN_DIR = "/content/drive/MyDrive/LLM_Assignment_3"
   %cd "$ASSIGN_DIR"
   ```

3. Open and run **`LLM_Assignment_3.ipynb`** top to bottom.  
   The notebook walks through:

   - Loading and preprocessing the Anthropic HH-RLHF dataset  
   - Training the reward model  
   - Training the SFT policy  
   - Training PPO, GRPO, and DPO policies  
   - Computing reward and KL statistics  
   - Saving generated samples and error cases

Trained models will be saved into the `models/` directory, and samples into `samples/`.

### 2. Running from the command line (local or Colab)

From the repo root:

```bash
cd src
```

You can run each stage separately using small one-liners.

#### 2.1 Train the reward model

```bash
python -c "from reward_model import train_reward_model; train_reward_model()"
```

This:

- Loads the HH-RLHF dataset
- Preprocesses chosen vs rejected pairs
- Trains the reward model
- Saves it to `../models/reward_model/` (plus `reward_head.bin`)

#### 2.2 Supervised fine-tuning (SFT)

```bash
python -c "from rl_algorithms import supervised_finetune; supervised_finetune()"
```

This trains a GPT-2-based policy on the **chosen** responses only and saves it as:

- `../models/policy_sft/`

#### 2.3 PPO training

```bash
python -c "from rl_algorithms import ppo_train; ppo_train()"
```

This:

- Starts from `policy_sft`
- Uses the reward model to score sampled responses
- Runs a small number of PPO updates
- Saves `../models/policy_ppo/`

#### 2.4 GRPO training

```bash
python -c "from rl_algorithms import grpo_train; grpo_train()"
```

This:

- Starts from `policy_sft`
- Samples groups of responses per prompt, computes group-relative advantages
- Updates the policy using a GRPO-style objective
- Saves `../models/policy_grpo/`

#### 2.5 DPO training

```bash
python -c "from rl_algorithms import dpo_train; dpo_train()"
```

This:

- Uses preference pairs `(prompt, chosen, rejected)` plus SFT as a reference policy
- Trains a DPO objective directly on log-prob differences
- Saves `../models/policy_dpo/`

#### 2.6 Optional: run all stages in sequence

If `train_all.py` is wired up to call each step, you can alternatively run:

```bash
python train_all.py
```

from inside `src/`.  
(This is optional and mainly exists to provide a single entry point.)

---

## Evaluation and outputs

### Reward model error analysis

- After training the reward model, we scan validation examples and collect cases where the reward model **prefers the rejected response** or ties.
- These examples are saved in:

  ```text
  analysis/reward_model_mistakes.jsonl
  ```

Each line is a JSON object with:

```json
{
  "idx": ...,
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "reward_chosen": ...,
  "reward_rejected": ...
}
```

This file is used in **ANLYSIS.md** to do qualitative error analysis of the reward model.

### Generated samples

We generate ~20 samples per model on a small set of prompts and save them as JSONL in `samples/`:

- `base_gpt2_samples.jsonl`
- `policy_sft_samples.jsonl`
- `policy_ppo_samples.jsonl`
- `policy_grpo_samples.jsonl`
- `policy_dpo_samples.jsonl`

Each line has the structure:

```json
{
  "prompt": "...",
  "response": "..."
}
```

These are used in ANLYSIS.md for qualitative comparison and failure mode discussion.

### Quantitative metrics

The notebook / evaluation code computes:

- Reward statistics (mean, std) for each model using the trained reward model
- Approximate KL divergence vs the SFT policy

Final numbers are summarized and interpreted in **`ANLYSIS.md`**.

---

## Notes on running environment

- All experiments were run on **Google Colab with a T4 GPU**, which is sufficient for:
  - GPT-2-sized models
  - Small-scale RL training (10 updates)
- Running everything on CPU is technically possible but will be **much slower**.
- For local execution, using a CUDA-capable GPU with at least ~8–12 GB of VRAM is recommended.

---

## Reproducibility

To roughly reproduce the reported results:

1. Ensure the Python and library versions are similar to Colab defaults (Python 3.10+, recent `transformers` and `datasets`).
2. Run the training stages in the order:
   1. `train_reward_model()`
   2. `supervised_finetune()`
   3. `ppo_train()`
   4. `grpo_train()`
   5. `dpo_train()`
3. Run the evaluation cells in the notebook (or equivalent scripts) to:
   - Evaluate reward and KL on test prompts
   - Save samples and error cases

Small differences in initialization or library versions can lead to slightly different metrics but the qualitative trends should remain similar.
