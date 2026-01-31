# Workable Test Cases

These are **manual** test cases you can run directly in the notebook. They are designed to be lightweight and unambiguous.

## Test Case 1 — Dataset preprocessing works
**Steps**
1. Run the “Prepare ToxiGen Dataset” section.

**Expected**
- Printed dataset structure shows only the `query` field.
- `dataset[0]` contains a non-empty `query` string (>= 10 words original).

---

## Test Case 2 — Model loads with 4-bit quant + tokenizer pads
**Steps**
1. Run “Initialize Model and Tokenizer”.

**Expected**
- No errors loading TinyLlama.
- `tokenizer.padding_side == "left"`
- `tokenizer.pad_token == tokenizer.eos_token`

---

## Test Case 3 — Reward function returns floats in [0,1]
**Steps**
1. Run “Define Reward Mechanism”.
2. Verify the printed “Reward function test”.

**Expected**
- Each sample text prints a numeric score, typically between 0 and 1.
- Higher-toxicity text tends to have higher score than benign text.

---

## Test Case 4 — PPO training produces histories
**Steps**
1. Run the PPO training section.

**Expected**
- `reward_history` and `loss_history` exist and have length > 0.
- Training plot renders without error.

---

## Test Case 5 — Evaluation generation + scoring runs end-to-end
**Steps**
1. Run the final evaluation cell (non-hate toxicity triggers).

**Expected**
- For each trigger, a “Generated Response” is printed.
- A “Reward Score” is printed for each.
- Outputs include profanity/harassment style language **without** protected-group hate.

---

## Optional Test Case 6 — Base vs trained comparison
**Steps**
1. In evaluation cell, set `use_base_model=True` to compare base model outputs (or rerun after clearing `actor`).
2. Compare reward scores and samples side-by-side.

**Expected**
- PPO-trained actor often yields higher reward scores and more consistently toxic style than base model.
