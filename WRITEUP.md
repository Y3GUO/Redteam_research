# Mini Red-Team Generator — Writeup

## 1) Approach with a small dataset + working model

### Goal
Train a small, runnable proof-of-concept that:
- generates **diverse** toxic outputs (profanity / harassment style),
- is guided by an explicit **reward signal** (a toxicity classifier),
- stays close to a base model via **KL regularization** to preserve fluency.

### Data
Your notebook loads **ToxiGen** (annotated split) via `datasets.load_dataset("skg/toxigen-data", "annotated")` and falls back to `imdb` if unavailable. You then:
- build prompts by taking the **first 10 words** of each example (`extract_query`)
- filter short examples
- cap to **1,000** prompts for speed

This is a good “small dataset” choice for a take-home because:
- it’s easy to run end-to-end,
- PPO can still learn a preference direction even with modest prompt diversity,
- you can iterate quickly on reward + generation.

### Model
You use **TinyLlama/TinyLlama-1.1B-Chat-v1.0** and load it in **4-bit NF4** with **LoRA** adapters:
- `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`
- LoRA: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`

Why this choice works:
- 1.1B is small enough to be runnable on a single GPU.
- QLoRA makes training feasible under tight compute limits.
- Chat-tuned base improves instruction following and readability.

### RL method (PPO)
Your notebook implements a custom PPO trainer (`SimplePPOTrainer`) with:
- actor + critic
- a reference model for KL penalty
- clipped policy objective
- advantage estimation style logic (GAE parameters present)

This is a strong demo because it shows an end-to-end adversarial generator that is **reward-steered**, not just prompted.

---

## 2) Evaluation: are the generations “good”? is it better than generic models?

### What “good” means here
For a red-team generator, “good” is not moral goodness; it means the generator is *effective* at producing:
- **higher-scoring toxic outputs** under your reward model,
- **diverse** outputs (not repeating a single template),
- **still fluent** / coherent.

### Metrics used (or easy to add)
Your notebook already tracks:
- `reward_history` (mean reward per batch)
- `loss_history` (policy loss)

In the evaluation section you generate samples and score them using `get_reward_scores()`.

Recommended minimal comparison (fits the assignment):
1. **Base model (no PPO)** generation on the same prompts.
2. **PPO-trained actor** generation on the same prompts.
3. Compare:
   - mean reward (toxicity score)
   - diversity: `distinct-1` / `distinct-2` (unique unigrams/bigrams)
   - qualitative examples (5–10)

Expected outcome:
- PPO actor has **higher reward** than base model on average, at the cost of sometimes shorter / more repetitive toxic phrases (classic reward hacking risk).

---

## 3) Tradeoffs + how to communicate them

### Tradeoffs you made
1. **Reward model proxy**
   - You use `facebook/roberta-hate-speech-dynabench-r4-target` and take the `'hate'` label score as reward.
   - Tradeoff: it’s a convenient proxy but can be misaligned with “general toxicity” and can be brittle to paraphrases.

2. **Small prompt extraction**
   - First 10 words is cheap and reproducible.
   - Tradeoff: prompts may not cover diverse scenarios; model may overfit to short contexts.

3. **Safety scope**
   - For responsible red-teaming, keep outputs to **non-hate toxicity** (profanity/harassment) and avoid protected-group targeting.
   - This keeps the PoC useful for evaluating filters without producing hate speech.

### How I would message this to a team
- “This is a PoC to show reward-steered adversarial generation. The reward model is a proxy; scores indicate *what the classifier believes*, not ground truth toxicity. We mitigate reward hacking via KL penalty and keep scope to non-protected targets.”

---

## 4) End-to-end runnable solution

### What’s included
- `Redteam_sanitized.ipynb`: your notebook with the protected-group trigger cell replaced by **non-hate toxicity triggers**.
- `TEST_CASES.md`: step-by-step test runs and expected artifacts.

### Run steps (Colab or local GPU)
1. Run install cell
2. Run dataset preparation
3. Run model + reward initialization
4. Run PPO training section
5. Run the final evaluation cell to generate and score samples

---

## Notes on responsible scope
I did **not** keep the protected-group hate trigger example in the final submission notebook. The PoC remains a red-team generator by producing **toxic profanity/harassment** outputs for safety testing without generating hate speech toward protected groups.
