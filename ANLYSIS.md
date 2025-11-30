# Assignment 3 - RLHF with PPO, GRPO, and DPO

## 1. Experimental setup

- **Base model:** GPT-2 (`gpt2`)
- **Dataset:** Anthropic HH-RLHF (`Anthropic/hh-rlhf`)
  - Train split used for reward model and policy training
  - Test split used for evaluation
- **Hardware:** Google Colab, T4 GPU
- **Core libraries:**
  - `torch`
  - `transformers`
  - `datasets`
  - `accelerate`

Pipeline:

1. Load and preprocess the HH-RLHF dataset
2. Train a reward model that scores chosen versus rejected responses
3. Train a supervised fine tuned policy (SFT) starting from GPT-2
4. Further train policies with PPO, GRPO, and DPO using the reward model and or preferences
5. Evaluate all models (base, SFT, PPO, GRPO, DPO) quantitatively with the reward model and KL, and qualitatively with generated samples

---

## 2. Dataset analysis and preprocessing (Part 1.1)

### 2.1 Dataset characteristics

The Anthropic HH-RLHF dataset provides triples `(prompt, chosen, rejected)`:

- `prompt` – user query or conversation context
- `chosen` – preferred assistant reply (helpful and harmless)
- `rejected` – dispreferred reply (less helpful, unsafe, low quality, etc.)

From manual inspection of random training examples:

- Prompts are often short, and sometimes empty in the processed split (the conversation context can be embedded within the replies themselves using "Human:" and "Assistant:" prefixes)
- Chosen responses:
  - More careful, on topic, and polite
  - Often emphasize safety, ethical constraints, and clear reasoning
- Rejected responses:
  - Often contain unsafe, speculative, or lower quality content
  - Sometimes look similar on the surface but miss safety details, hedging, or nuance

Overall, the dataset encodes a notion of good assistant behavior that blends helpfulness and harmlessness.

### 2.2 Preprocessing pipeline

All preprocessing is implemented in `src/data_utils.py`.

**Tokenizer**

- Tokenizer: `AutoTokenizer.from_pretrained("gpt2")`
- GPT-2 has no pad token, so we set:
  - `tokenizer.pad_token = tokenizer.eos_token`

**Reward model inputs**

For each pair `(prompt, chosen, rejected)` we build:

```text
text_chosen   = prompt + "\n\nAssistant: " + chosen
text_rejected = prompt + "\n\nAssistant: " + rejected
```

Then we tokenize each with:

```python
tokenizer(
    text,
    max_length=512,
    truncation=True,
    padding="max_length",
)
```

The resulting dataset contains:

- `input_ids_chosen`, `attention_mask_chosen`
- `input_ids_rejected`, `attention_mask_rejected`

We shuffle and split into train and validation with a 90 percent and 10 percent split.

**SFT and policy inputs**

For SFT we construct:

- Prompt tokens
- Answer tokens (chosen)
- A concatenated sequence where:
  - Prompt tokens remain in the context
  - Loss is only applied to answer tokens, using labels equal to `-100` on prompt positions

For RL methods, prompts are tokenized and used as context for sampling. Generated responses are then re encoded to compute log probabilities and reward model scores.

---

## 3. Reward model training and error analysis (Part 1.2)

### 3.1 Reward model architecture and training

**Architecture**

- Backbone: `AutoModelForCausalLM.from_pretrained("gpt2")`
- Reward head:
  - Take the final hidden state corresponding to the last token
  - Feed it through a linear layer `hidden_size -> 1` to obtain a scalar reward

Let:

- `r_c = reward(chosen)`
- `r_r = reward(rejected)`

We optimize a pairwise preference loss:

\[
L = - \log \sigma(r_c - r_r)
\]

which encourages `r_c > r_r`, so chosen responses receive higher scores.

**Training details**

- Mini batch size: small (around 4) to fit on T4
- Learning rate: `1e-5`
- Epochs: 1 over a subset of the train split (for speed)
- Device: GPU

During training we log:

- Per step loss and accuracy on mini batches
- Validation accuracy on the held out validation portion

**Validation accuracy**

From training logs:

- Validation accuracy is approximately 0.43

That is, on about 43 percent of validation pairs the reward model correctly assigns a higher score to the chosen response than to the rejected one. This is somewhat better than chance, but clearly imperfect and noisy as a preference oracle.

### 3.2 Qualitative error analysis

We collected 20 mistake cases where:

\[
r_{	ext{chosen}} \le r_{	ext{rejected}}
\]

and inspected each prompt, chosen, rejected triple (from `reward_model_mistakes.jsonl`).

Some notable patterns:

#### Pattern 1: preference for more concrete but less careful answers

In scientific or explanatory questions (for example, "Why did cells originally combine together to create life"), the chosen answers tend to:

- Explicitly acknowledge uncertainty
- Use hedging language such as "scientists believe" or "one hypothesis is"
- Emphasize that our understanding is incomplete

The rejected answers sometimes:

- Give an over confident, simplified story that sounds decisive
- Focus on concrete benefits such as "cells cooperate to reduce competition"

In several mistakes, the reward model assigns a higher score to the over confident rejected response, likely because it is more direct and assertive, even if epistemically weaker.

#### Pattern 2: sensitivity to stylistic differences (politeness, verbosity)

In recommendation style prompts such as "suggest a movie" or "host guests and cook Hong Kong dishes":

- Chosen responses are usually longer, more detailed, and polite (for example, "I would recommend X because Y")
- Rejected responses may be shorter or more blunt

The reward model sometimes prefers the rejected answer when it is:

- Slightly shorter
- More punchy or strongly opinionated

even though the chosen answer better matches the intended politeness and carefulness of the dataset. This suggests the model partially captures style, but not always the correct stylistic preferences.

#### Pattern 3: safety blind spots

Some prompts are safety critical, for example:

> "Tell me how to steal an iPhone."

Here:

- The chosen response tries to avoid giving harmful instructions or redirects the user away from crime
- The rejected response continues the conversation and suggests clearly harmful or illegal behavior (for example, exploiting sellers or setting up fake accounts)

In at least one such case, the reward model scores are roughly:

- `reward_chosen ≈ 3.92`
- `reward_rejected ≈ 7.33`

So the reward model gives a much higher score to the harmful rejected response. This shows a serious limitation:

- The reward model is not reliably capturing safety
- It sometimes rewards longer and more "useful looking" answers even when they are unethical

#### Pattern 4: ambiguity and near duplicates

Some mistake pairs have chosen and rejected responses that are very similar in wording and length. In these cases:

- The distinction between "chosen" and "rejected" is subtle
- The reward model errors are less serious because the two responses are qualitatively close

However, they still show that the preference signal is noisy, and small stylistic differences can flip the sign of `r_c - r_r`.

#### Summary of reward model behavior

- The reward model usually prefers safer and more on topic responses
- It sometimes overweights assertiveness, concreteness, and length
- It has clear safety blind spots where harmful answers are rewarded more highly
- It is noisy when differences are stylistic or subtle

This must be kept in mind when using this model as the objective for PPO and GRPO training. Heavy optimization risks reward hacking and misalignment.

---

## 4. Policy training: SFT, PPO, GRPO, DPO (Parts 2 and 3)

### 4.1 Supervised fine tuning (SFT)

We first train a supervised policy `policy_sft`:

- Initialization: GPT-2 (`gpt2`)
- Objective:
  - Standard causal language modeling loss on tokens belonging to the chosen response
  - Prompt tokens are included in the input but masked with `-100` in the labels
- Hyperparameters (approximate):
  - Batch size: 2
  - Learning rate: `1e-5`
  - Epochs: 1
  - Max sequence length: 512

`policy_sft` serves two roles:

1. A stronger baseline than base GPT-2
2. A reference policy used for
   - KL comparison in PPO and GRPO analysis
   - The reference distribution in DPO

### 4.2 PPO

PPO training is implemented in `ppo_train`:

- Initialization: `policy_sft`
- For each update:
  1. Sample responses for a prompt using a custom manual sampling loop (no `model.generate`)
  2. Compute:
     - `log pi_new` under the current policy
     - `log pi_old` detached (for the ratio)
     - `log pi_ref` under the SFT policy (for KL)
  3. Compute reward for each response with the reward model
  4. Use the reward as a simple advantage
  5. Optimize the clipped PPO objective plus a KL penalty toward SFT

Key hyperparameters:

- Learning rate: `1e-5`
- Number of updates: 10
- Max sampled tokens: 32
- KL coefficient: 0.1
- Clip range: 0.2

### 4.3 GRPO

GRPO training is implemented in `grpo_train`:

- Initialization: `policy_sft`
- For each update:
  1. Sample a group of size `K = 4` responses for one prompt
  2. Score each response with the reward model
  3. Compute group mean reward and define advantages
     - `advantage_i = r_i - mean_group_reward`
  4. Use a REINFORCE style objective
     - `loss = - E[advantage * log pi(response)]`

This encourages responses that are better than their group peers and discourages worse ones.

Hyperparameters:

- Learning rate: `1e-5`
- Number of updates: 10
- Group size: 4
- Max generated tokens: 32

### 4.4 DPO

DPO training is implemented in `dpo_train`:

- Initialization: `policy_sft`
- Uses preference pairs `(prompt, chosen, rejected)` plus a fixed reference policy (SFT)

For each pair:

- Compute sequence log probabilities:
  - `log pi(c)`, `log pi(r)` under the current policy
  - `log pi_ref(c)`, `log pi_ref(r)` under SFT

Define:

\[
\Delta_{\pi} = \log \pi(c) - \log \pi(r), \quad
\Delta_{	ext{ref}} = \log \pi_{	ext{ref}}(c) - \log \pi_{	ext{ref}}(r)
\]

The DPO loss is

\[
L = - \mathbb{E} \left[ \log \sigma\left( eta (\Delta_{\pi} - \Delta_{	ext{ref}}) 
ight) 
ight]
\]

where `beta = 0.1` controls strength of alignment to the preferences versus the reference.

Hyperparameters:

- Batch size: 2
- Learning rate: `1e-5`
- Epochs: 1
- Beta: 0.1
- Max length: 512

---

## 5. Quantitative evaluation (Part 4.1)

We evaluate the following models on 100 held out test prompts:

- `base_gpt2`
- `policy_sft`
- `policy_ppo`
- `policy_grpo`
- `policy_dpo`

For each prompt:

1. Generate a response with each model
2. Compute its reward using the trained reward model
3. Compute an approximate KL like quantity versus SFT

\[
	ext{KL approx} pprox \mathbb{E}[\log P_{	ext{model}} - \log P_{	ext{SFT}}]
\]

on the generated sequence.

### 5.1 Reward statistics

From the evaluation code:

```text
=== Reward stats (mean, std) ===
base_gpt2   : mean=5.023, std=1.568
policy_sft  : mean=5.407, std=1.341
policy_ppo  : mean=5.498, std=1.576
policy_grpo : mean=5.578, std=1.207
policy_dpo  : mean=5.073, std=2.299
```

This gives:

| Model        | Mean reward | Std reward |
|-------------|------------:|-----------:|
| base_gpt2   | 5.023       | 1.568      |
| policy_sft  | 5.407       | 1.341      |
| policy_ppo  | 5.498       | 1.576      |
| policy_grpo | 5.578       | 1.207      |
| policy_dpo  | 5.073       | 2.299      |

Observations:

- All trained policies, except possibly DPO, improve over base GPT-2 in terms of mean reward
- SFT already gives a clear gain over base GPT-2
- GRPO has the highest mean reward (5.578) and the lowest standard deviation (1.207) among RL methods, suggesting consistently strong performance under this reward model
- PPO also improves reward (5.498) but has variance similar to base GPT-2
- DPO has mean reward only slightly above base GPT-2 (5.073 vs 5.023) but with larger variance (2.299), indicating more unstable behavior under this reward model

Because the reward model is imperfect, these numbers should be interpreted as alignment to the reward model, not necessarily alignment to true human preferences.

### 5.2 Approximate KL versus SFT

From the same evaluation:

```text
=== Approx KL vs SFT (E[log P - log Pref]) ===
base_gpt2   : KL≈-0.0871
policy_ppo  : KL≈0.5407
policy_grpo : KL≈0.0112
policy_dpo  : KL≈0.0578
```

(For `policy_sft` the KL versus itself is zero by definition.)

We can summarize:

| Model        | Approx KL vs SFT (E[log P - log Pref]) |
|-------------|-----------------------------------------|
| base_gpt2   | -0.0871                                 |
| policy_ppo  |  0.5407                                 |
| policy_grpo |  0.0112                                 |
| policy_dpo  |  0.0578                                 |

Interpretation:

- Base GPT-2 has a slightly negative value, meaning that on these sequences SFT assigns higher likelihood than base GPT-2, which is expected
- PPO deviates the most from SFT (around 0.54). It pushes the policy further away in log probability space, consistent with aggressive reward optimization
- GRPO has very small KL (around 0.01), so it stays extremely close to SFT while still improving reward the most
- DPO has a small positive KL (around 0.06), so it moves somewhat away from SFT but less than PPO

### 5.3 Pareto trade off

Considering both mean reward and approximate KL:

- PPO: high reward but also large KL, which increases risk of reward hacking and distribution shift from the supervised baseline
- GRPO: highest reward with almost no KL increase, which looks close to a Pareto efficient point in this small experiment
- DPO: small KL and small reward gain, modestly improving preference alignment while staying near SFT
- Base GPT-2: lowest reward and negative KL, farthest from SFT and clearly worst overall

---

## 6. Qualitative evaluation and failure modes (Part 4.2)

### 6.1 Samples on generic prompts

We generated around 20 samples per model and saved them as:

- `base_gpt2_samples.jsonl`
- `policy_sft_samples.jsonl`
- `policy_ppo_samples.jsonl`
- `policy_grpo_samples.jsonl`
- `policy_dpo_samples.jsonl`

Many prompts in this slice are empty or very short, so the models effectively free run from a conversational prior (for example starting with "Human: Hello" and "Assistant:").

Typical qualitative behavior:

- **Base GPT-2**
  - Responses often drift into incoherent or chaotic stories with multiple voices and inconsistent narrative
- **SFT**
  - Responses are more coherent and polite, with clearer assistant style behavior
- **PPO**
  - Responses tend to be longer, with emotionally intense or dramatic language
  - Content sometimes looks like it is optimized for reward model preferences rather than for human readability
- **GRPO**
  - Responses remain close in tone and style to SFT
  - Often slightly more detailed or structured, but not as extreme as PPO
- **DPO**
  - Responses broadly resemble SFT but can be more variable, with some generations drifting into odd narrative content

From these samples:

- SFT is clearly better than base GPT-2 in politeness and coherence
- PPO can appear to over optimize the reward model, producing long and somewhat unnatural outputs
- GRPO matches the quantitative picture, staying close to SFT while improving reward
- DPO feels similar to SFT but with more variance from sample to sample

### 6.2 Adversarial and failure mode prompts

Using the insights from reward model analysis, we can reason about likely failure modes:

1. **Safety critical prompts**

   Because the reward model sometimes scores harmful rejected answers higher than safe chosen answers, both PPO and GRPO could in principle be pushed toward unsafe behavior if trained longer or more aggressively on this reward. Our small scale runs do not fully exploit this, but the risk is clearly present.

2. **Over agreement and sycophancy**

   PPO often produces very confident, long answers that strongly buy into the user framing. This can be problematic on questions that require skepticism or refusal. This is consistent with PPO having high KL relative to SFT.

3. **Out of distribution nonsense**

   For weird or empty prompts, base GPT-2 and RL trained policies can drift into nonsense. SFT and GRPO tend to stick closer to an assistant style, while PPO and DPO sometimes wander further into strange narrative content.

### 6.3 Which model feels best

From these small experiments:

- Base GPT-2 is clearly worst: lowest reward and poorest qualitative behavior
- SFT is already a strong baseline: better reward and much more "assistant like"
- PPO improves reward but shows signs of reward hacking, with overly dramatic or verbose outputs
- GRPO appears to be the best trade off:
  - Highest mean reward
  - Very low KL relative to SFT
  - Qualitative behavior that stays close to SFT but slightly enhanced
- DPO is stable and simple to train, with behavior similar to SFT and modest reward gains

---

## 7. Discussion and conclusion

### 7.1 Main takeaways

1. **Reward model is the main bottleneck**

   With validation accuracy around 0.43 and clear safety blind spots, the reward model is far from a perfect proxy for human preferences. Strong RL optimization against it risks reward hacking and misalignment.

2. **SFT already gives a strong boost**

   Training GPT-2 on chosen responses yields a policy that:
   - Outperforms base GPT-2 on the reward metric
   - Produces more assistant like responses in practice

3. **PPO, GRPO, and DPO each have different trade offs**

   - PPO:
     - Achieves higher reward than SFT
     - Introduces the largest KL shift from SFT
     - Sometimes produces qualitatively strange or overly dramatic outputs
   - GRPO:
     - Achieves the highest reward in this experiment
     - Keeps KL almost unchanged relative to SFT
     - Qualitatively stays close to SFT while being more reward aligned
   - DPO:
     - Gives small reward gains and small KL shifts
     - Behavior is similar to SFT but with higher variance

### 7.2 If I had to ship one model

Given:

- The noisy and imperfect reward model
- The trade off between reward and KL
- The observed qualitative behavior

The most reasonable choice in this setting would be **GRPO**:

- It achieves the highest reward under the reward model
- It remains extremely close to SFT in distribution
- It appears less prone to extreme behavior than PPO

In a real deployment, the right path would be:

1. Improve the reward model, especially on safety
2. Use human evaluation or a stronger LLM judge for final model comparison
3. Consider SFT plus a small amount of RL (PPO or GRPO) with strong monitoring, instead of aggressive optimization against a weak reward model
