# Phase 1 Data Model: Label Warm-Start with P&L Fine-Tune

**Feature**: 010-labelfree-warmstart-finetune · **Date**: 2026-09-11

Entities extend brief 01's rather than replacing them. Fields already defined there
(TrialRecord's seed/fold/cost-scale, SimulatedRun, Fold) are inherited and only the additions
are documented.

---

## 1. ArmConfig

One experimental treatment. Arms differ **only** in initialisation and objective; every other
component is shared with brief 01 (FR-002).

| Field | Type | Notes |
|---|---|---|
| `arm` | enum | `warm_start` \| `blended` \| `auxiliary` |
| `init` | enum | `classifier` (Arm A) \| `random` (B, C) |
| `warm_start_ref` | string \| null | Identity of the per-fold classifier whose encoder was inherited; null for B and C |
| `encoder_lr`, `head_lr` | float | Arm A requires `encoder_lr` an order of magnitude below `head_lr` (FR-005) |
| `blend_alpha` | float \| null | Arm B only; one of {0.75, 0.5, 0.25} |
| `label_threshold` | float \| null | Arm B only; 0.007 (production) or 0.02 |
| `aux_target` | enum \| null | Arm C only; `forward_realised_vol` |
| `purge_bars`, `embargo_bars` | int | Actual values applied, not requested (FR-014) |

**Validation**
- `warm_start_ref` MUST be non-null exactly when `init = classifier`.
- Arm A MUST satisfy `encoder_lr * 10 <= head_lr`.
- `blend_alpha` and `label_threshold` MUST be null outside Arm B, so a stray label term cannot
  enter another arm unnoticed.
- Any arm with a forward-looking term MUST have `purge_bars > 0` and `embargo_bars > 0`; a run
  with either at zero is void, not caveated (FR-014, SC-012).

---

## 2. WarmStartSource

The classifier whose encoder Arm A inherits.

| Field | Type | Notes |
|---|---|---|
| `fold_id` | string | The fold this classifier was trained for |
| `train_range` | [date, date] | MUST equal the fold's own training range |
| `weights_ref` | string | Path to the saved classifier weights |
| `architecture` | object | `{resize, conv1, ln_conv, lstm, ln_lstm, num_filters}` |
| `label_threshold` | float | The label the classifier was trained on |
| `inherited_tensors` | list[string] | Which tensors actually transferred |

**Validation**
- `train_range` MUST equal the fold's training range. **The production model MUST NOT be used**
  — it was trained on data later folds treat as held out, so inheriting it leaks the test range
  into the initialisation (FR-004).
- `inherited_tensors` MUST be non-empty and MUST exclude the classifier's classification head:
  only the encoder transfers.
- Post-load, the policy's encoder weights MUST differ from a randomly-initialised policy of the
  same architecture — otherwise nothing was inherited and the arm is mislabelled.

---

## 3. LabelTarget

A forward-looking training target. Both the directional label and the auxiliary volatility
target are instances, so both inherit the same leakage controls.

| Field | Type | Notes |
|---|---|---|
| `kind` | enum | `directional` (Arm B) \| `forward_realised_vol` (Arm C) |
| `horizon_bars` | int | The label horizon; also the purge and embargo length |
| `threshold` | float \| null | Directional only |
| `values` | series[bar, pair] | Aligned to the panel's shared date index |
| `valid_mask` | series[bar, pair] | False where the target cannot be computed causally |

**Validation**
- Values within `horizon_bars` of a fold boundary MUST be purged.
- `forward_realised_vol` MUST carry no directional information — it is a magnitude, so its sign
  must be constant by construction.
- The target MUST align to the same date index as the feature panel; positional alignment is
  prohibited (inherited from 009).

---

## 4. LossBreakdown

Recorded per training step so a nominal blend that has collapsed to one term is detected rather
than assumed balanced (FR-009).

| Field | Type | Notes |
|---|---|---|
| `profit_term`, `label_term`, `aux_term` | float | Realised contribution of each |
| `participation_penalty` | float | Inherited degeneracy guard |
| `alpha_nominal` | float | The configured blend weight |
| `alpha_realised` | float | `label_term / (label_term + profit_term)` |

**Validation**
- `alpha_realised` MUST be reported beside `alpha_nominal`. A blend whose realised share is near
  0 or 1 is not a blend, and reporting only the nominal weight would hide that.
- Every term MUST be finite; a non-finite loss aborts the run (inherited from 009).

---

## 5. ArmResult

Extends brief 01's trial result.

| Field | Type | Notes |
|---|---|---|
| `test_log_return`, `test_mean_exposure`, `test_turnover`, `test_flat_share` | float | As brief 01 |
| `exposure_label_corr` | float \| null | Arm A: correlation between realised exposure and the inherited label (FR-006). **Diagnostic only — never an acceptance criterion** |
| `alpha_realised` | float \| null | Arm B |
| `degenerate` | bool | True when mean exposure is below the participation floor |

**Validation**
- `degenerate = True` MUST exclude the configuration from carry-forward and MUST be reported as
  inconclusive rather than as a return (FR-020, SC-015).
- `exposure_label_corr` near 1 is recorded as "the fine-tune changed nothing", near 0 as "the
  prior was discarded". Both are findings; neither decides acceptance (Principle III).

---

## 6. ResultReport

The study's per-brief write-up, extending brief 01's required content with:

- The **directional finding**: whether labels helped or hurt relative to the P&L control,
  recorded regardless of whether any arm is accepted (FR-022, SC-009).
- **Both P&L control figures**: the control re-run under this architecture, and brief 01's
  original, so the architecture effect is visible separately from the label-prior effect
  (FR-004b, SC-010a).
- Per-arm mean exposure beside every return, so a degenerate arm cannot rank on inaction.
