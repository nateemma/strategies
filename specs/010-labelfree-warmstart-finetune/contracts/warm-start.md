# Contract: Warm-start (Arm A)

## Source

A classifier trained on **the fold's own training range**. The production model is never used:
it was trained on data that later folds treat as held out, so inheriting it leaks the test range
into the initialisation — the same class of error as ranking on the training range, which cost
feature 009 a full run.

## Transfer

The classifier exposes `encode()` and `decode()`. Arm A inherits `encode()` — `resize`, `conv1`,
`ln_conv`, `lstm`, `ln_lstm` — and replaces `decode()`, which ends in a 3-class softmax, with the
exposure head. The classification head is never inherited.

```
inherited : resize, conv1, ln_conv, lstm, ln_lstm      -> latent (B, num_filters)
new       : exposure head (+ optional no-trade band)   -> w in [0,1]
```

## Required checks

| Check | Why |
|---|---|
| Inherited tensor names and shapes match the policy's encoder exactly | A silent partial load leaves most of the encoder random and mislabels the arm |
| Post-load encoder weights differ from a randomly-initialised policy of the same architecture | Proves something was actually inherited |
| The classification head is absent from the inherited set | Only the representation transfers, not the label decision |
| `encoder_lr * 10 <= head_lr` | FR-005 |
| The source classifier's training range equals the fold's | FR-004 |

## Diagnostic, not a gate

The correlation between the policy's realised exposure and the label it inherited is reported
(FR-006, SC-011): near 1 means the fine-tune changed nothing, near 0 means the prior was
discarded. Both are informative. **Neither decides acceptance** — Principle III forbids a
model-fit quantity from deciding a verdict, and this is one.
