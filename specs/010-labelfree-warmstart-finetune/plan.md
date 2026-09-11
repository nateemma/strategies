# Implementation Plan: Label Warm-Start with P&L Fine-Tune

**Branch**: `010-labelfree-warmstart-finetune` | **Date**: 2026-09-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/010-labelfree-warmstart-finetune/spec.md`

## Summary

Brief 01 could not distinguish "the profit objective is wrong" from "gradient descent from
random initialisation cannot find a trading policy at this signal-to-noise ratio". This feature
separates them by treating the repository's existing lookahead labels as a *prior* rather than a
target, across three arms: warm-start from a per-fold classifier and fine-tune on profit (A), a
blended label/profit objective from random init (B), and profit plus a non-directional auxiliary
head (C).

Technical approach: all three arms use the classifier's encoder architecture, so Arm A's
warm-start is a literal weight inheritance rather than a lossy projection. The classifier already
exposes `encode()`/`decode()`, so the exposure head replaces `decode()` with no surgery and
brief 01's simulator, feature pipeline, trial log and wrapper are reused unchanged. Both controls
— the production classifier and brief 01's best trading configuration — are evaluated, with the
P&L control re-run under this architecture so the label-prior effect is not confounded with the
architecture change.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)

**Primary Dependencies**: MLX (`nn.LSTM`, `nn.Conv1d`, `nn.LayerNorm`, `nn.Linear`); the existing
`NNNClassifierMLX_LSTM` encoder; brief 01's `research/labelfree/` modules (simulator, features,
folds, costs, trials, stats, train); `regime/wf/` for pooled evaluation and comparison;
`Framework/TrainingSignals.py` for the labels.

**Storage**: Feather OHLCV under `user_data/data/binanceus/`; per-fold classifier weights and
policy weights under `saved_data/`; the shared append-only trial log at
`research/labelfree/trials.jsonl`.

**Testing**: Value-asserting regression tests alongside brief 01's 16; a leakage test that the
purge and embargo are applied; a warm-start test that inherited weights actually load and differ
from random initialisation.

**Target Platform**: macOS on Apple Silicon (Metal GPU), single machine, sequential execution.

**Project Type**: Research feature — three training-arm configurations plus evaluation. No new
service, no UI, no new strategy family beyond brief 01's wrapper.

**Performance Goals**: The windowed encoder is **0.6× the cost** of brief 01's streaming pass
(measured), so per-run training time is not expected to exceed brief 01's ~200 s. The dominant
new cost is training one classifier per fold for Arm A — one for the screen, fifteen for a
pooled run.

**Constraints**: Window tensors scale as `bars × seq_len × features × pairs` and reach **2.14 GB**
on the full train panel — they MUST be built per chunk (0.04 GB) and never for the whole panel.
Every arm with a forward-looking term purges the label horizon and embargoes one horizon.
Sequential runs only. Only this feature's files are staged.

**Scale/Scope**: 10-pair walk-forward universe, 15-minute bars, 15 anchored folds. Screen on f08.
Arm A: 1 config × 5 seeds. Arm B: 3 blend weights × 2 labels × 5 seeds = 30 runs. Arm C: 1 × 5.
Plus the P&L control re-run under the new architecture.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Gate | Pre-Phase-0 | Post-Phase-1 |
|---|---|---|---|
| I. Evaluate in the deployment regime | Windows from the protocol | PASS | PASS |
| II. Persistence across distant eras (NON-NEGOTIABLE) | A distant era must be tested | PASS — SC-002 requires all three windows | PASS |
| III. Learnability is not edge | No model-fit metric may decide acceptance | PASS — every SC is cost-aware P&L; the exposure/label correlation (FR-006, SC-011) is explicitly a **diagnostic, not an acceptance criterion** | PASS |
| IV. Verify parameters are actually live | Changed behaviour measured downstream | PASS — FR-009 measures each loss term's realised contribution rather than assuming the blend is balanced | PASS |
| V. Lookahead-clean is not realistic | Realistic execution mandatory | PASS — inherited calibrated simulator; FR-013/014 purge and embargo | PASS |
| VI. Diagnose before tuning | Diagnostics before hyperparameters | PASS — US1 scenario 4 records an underperforming Arm A and moves on rather than tuning it | PASS |
| VII. Validate cheaply before building | Cheap gate before the expensive build | PASS — screen on f08 before any pooled run or 15-fold classifier training | PASS |
| VIII. One parameter space at a time | No joint search on thin data | PASS — arms are separate; only the blend weight is swept, and FR-010 forbids reporting a best value | PASS |
| IX. Orthogonality | Factor correlation first | N/A — single policy | N/A |
| X. Structural fixes, not per-pair overrides | No pair-specific code | PASS — Non-Goals bars it | PASS |
| Repository boundary / staging | Outer checkout untouched; stage own files | PASS — FR-024 | PASS |
| Framework stays strategy-agnostic | No strategy code in `Framework/` | PASS — new code lives in `research/labelfree/`; the classifier encoder is imported, not moved | PASS |
| Workflow gates 1–7 | Pinned ranges, distant era, realistic execution, liveness, seed robustness, trade count, scalers | PASS — FR-013/016/019/020; scalers unchanged from 009 | PASS |

**Result: no unjustified violations.** Note that feature 009's Principle VIII exception does not
carry over — this feature sweeps one parameter (the blend weight) within one arm, not a
factorial across three dimensions.

## Project Structure

### Documentation (this feature)

```text
specs/010-labelfree-warmstart-finetune/
├── plan.md              # This file
├── spec.md              # 28 FR / 17 SC, 3 clarifications resolved
├── research.md          # Phase 0 — 9 decisions with measurements
├── data-model.md        # Phase 1 — entities and validation rules
├── quickstart.md        # Phase 1 — runnable validation guide
├── contracts/           # Phase 1 — arm and warm-start contracts
│   ├── arms.md
│   └── warm-start.md
├── checklists/
│   └── requirements.md  # 16/16
└── tasks.md             # Phase 2 — /speckit-tasks, NOT this command
```

### Source Code (repository root = `user_data/strategies/`)

```text
research/labelfree/                 # EXISTING from brief 01 — reused, not forked
├── sim.py, costs.py, folds.py      # unchanged
├── features.py, trials.py          # unchanged (sequence_tensor already builds windows)
├── stats.py                        # unchanged
├── train.py                        # EXTENDED: arm config, warm-start init, blended loss
├── losses.py                       # EXTENDED: blended CE+utility, auxiliary volatility loss
├── policy.py                       # EXTENDED: encoder variant matching the classifier
├── labels.py                       # NEW — label + forward-volatility targets, purge/embargo
├── warmstart.py                    # NEW — per-fold classifier training and weight inheritance
├── arms.py                         # NEW — A/B/C as configurations of the existing trainer
├── baselines.py                    # NEW — buy-and-hold + pinned 20/60 EMA trend rule (SC-003).
│                                   #   Brief 01 planned it; its Phase 6 was unreached, so it
│                                   #   does not exist and must be built here
├── reproduce.py                    # NEW — fresh-process reproduction from a saved config (SC-006).
│                                   #   Same status: planned by brief 01, never built
└── tests/
    ├── test_warmstart.py           # NEW — weights load and differ from random init
    ├── test_purge_embargo.py       # NEW — leakage control actually applied
    └── (16 existing tests)         # unchanged

NNNC/NNNClassifierMLX.py            # imported read-only for its encode(); NOT modified
NNDP/NNDP_MLX.py                    # brief 01's wrapper, reused
research/label-free-study/briefs/results/02-<date>.md   # NEW — protocol §6 report
```

**Structure Decision**: Everything new lives in `research/labelfree/` beside brief 01's code,
because the arms are configurations of that trainer rather than a parallel implementation — which
is what FR-001/FR-002 require and what makes differences between arms attributable to
initialisation and objective. The classifier is *imported* for its `encode()`; it is not copied,
moved, or modified, so `Framework/` and `NNNC/` stay untouched.

## Complexity Tracking

> Risks recorded so they are not discovered late. None is a constitution violation.

| Item | Why accepted | Simpler alternative rejected because |
|------|--------------|--------------------------------------|
| All arms change architecture relative to brief 01 | Required for the warm-start to be a literal weight inheritance rather than a lossy projection, which is the only form whose null result is interpretable | Projecting LSTM(64) into LSTM(32) preserves comparability but makes a null uninterpretable — you could not tell whether the prior failed or the projection destroyed it |
| The P&L control must be re-run, not cited | Comparing against brief 01's recorded numbers would confound the label prior with the architecture change, and isolating the prior is the entire purpose | Citing brief 01's figures is free but answers a different question than the one asked |
| The P&L control loses 7.06% out of sample | It is the only brief-01 arm that genuinely trades, so it is the only control against which "did labels help" is answerable | The top-*ranked* brief-01 configs are flat on 100% of bars; any arm that trades beats them trivially, making SC-009 vacuous |
| Arm A needs 15 classifiers for a pooled run | Warm-starting from the production model would leak later folds' test ranges into the initialisation — the same class of error as ranking on the training range | Reusing one model is far cheaper and is exactly the leak the per-fold rule exists to prevent |
| Two modules brief 01 planned were never built | `baselines.py` and `reproduce.py` were in brief 01's structure tree but its Phases 6-7 were unreached, so they do not exist. This feature needs both — they carry SC-003 and SC-006 — and builds them | Assuming they existed is exactly the latent-dependency failure that surfaced in brief 01 as a task claiming to have run an `audit_trials.py` that was never written. Caught here before implementation rather than after |
| Both labels tested despite a recorded null | The repo measured 0.007 vs 0.02 as identical in P&L (332 trades, 10.45% both) but 38% apart in label count; as a *prior* rather than a target, density may matter where P&L did not | Testing one label saves a sweep, but the brief asks for both and the asymmetry is cheap to settle |
