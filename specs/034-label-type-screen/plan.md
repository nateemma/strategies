# Implementation Plan: Label-Type Screen for NNNC

**Branch**: `034-label-type-screen` (spec directory; committed to master) | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/034-label-type-screen/spec.md`

## Summary

Train ARO's architecture on seven other label types (early folds only), reuse the existing gbb_path
(control) and gbb_fwdmae (GFM) weights, and ask of each: does it rank width-book trades better than
gbb_path? The screen runs in stages:

1. **S0a label audit** (no model): class shares after peak filter and conflict resolution,
   parameter liveness, feature-list identity. Degenerate candidates stop here, before training.
2. **S0** after training: weights, keys, label–width Spearman.
3. **S1 separation** (no backtest): agree − disagree gap on identical width-book trades, against
   both the pre-registered 0.072 and the control at the same 0.5 agree share.
4. **S2**: up to 3 survivors backtested on f00–f07 as a width-book sizer and as an ungated entry
   source (rate-matched to the control), against the width book and the control.
5. **S3**: the best S2 arm trained and judged on f08–f14; on a pass, GBR repeats it.

**Technical approach**: no Framework/ change. Generated NNNC classes per candidate
(`NNNC_LS_<TAG>` and sizer twin `_SZ`, same model key), a candidate-aware generalisation of 033's
pick / separation / judge scripts in `research/label_screen/`, and runtime registration of the new
plans in the shared signal-table and training drivers (R3–R8).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**: freqtrade (backtesting, strategy resolver), MLX (LSTM_RAW), numpy,
pandas, scipy (Spearman, AUC)

**Storage**:
- weights at new 030 model-key locations (one key per candidate and fold);
- plans and thresholds in `evaluation/folds/` (`plan_ls_<tag>_sz.json`, `plan_ls_<tag>_es.json`);
- signal tables in `research/nnnc_framework/signals/LS_<TAG>_fNN.parquet`;
- arm records in `regime/arms/`;
- study outputs in `research/label_screen/`.

**Testing**: pytest, new `research/label_screen/tests/`; the full suite (currently 994 passed,
9 skipped) must stay green.

**Target Platform**: macOS on Apple Silicon, local.

**Project Type**: research framework (nested strategies repository).

**Performance Goals**:
- training about 12 min per ARO fold: 7 × 8 ≈ 11 h, one sequential chain;
- 56 signal-table passes and 56 single-fold sizer picks;
- S2: at most 3 survivors × 2 roles × 8 folds of backtests (about 13 s each), plus up to 24
  entry-rate picks;
- conditional: a seed-2 repeat (≈ 1.6 h per label on f00–f07, plus ≈ 1.6 h for the ARO control at
  seed 2, which has no records yet), the S3 winner on f08–f14 (≈ 1.4 h) and the GBR repeat
  (15 folds, ≈ 0.5 h).

**Constraints**: one chain at a time; picks one fold per process; byte-identical when off;
Framework/ strategy-agnostic; live bot untouched; no default promoted.

**Scale/Scope**: ARO, 9 labels (7 trained), early folds for the screen; late folds and GBR only for
the S2 winner.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / constraint | Status | How |
|---|---|---|
| I. Deployment regime | PASS | 2023–26 walk-forward; late half confirmatory and read only for the winner. |
| II. Distant eras | PASS by restriction | No distant era for NNNC models: at most PASS (PROVISIONAL), with a forward-confirmation route. |
| III. Learnability ≠ edge | PASS | S1 separation is measured on realised trade pp, and the verdict needs backtest P&L (S2/S3) against the width book and control. Label learnability is not a gate. |
| IV. Parameters live | PASS | S0a parameter-liveness diff per declared extra; 4-decimal threshold load check; tag-to-agree check; rate-match check for entry-source arms. |
| V. Realistic execution | PASS | Same config, fees and liquidity cap; realised stake ratios and wallet shortfalls reported. |
| VI. Diagnose before tuning | PASS | S0a records each label's class balance and sell-side behaviour before training (R1); no tuning at all. |
| VII. Validate cheaply | PASS | S0a (labels only) and S1 (no backtest) gate the expensive steps; an f00 smoke precedes the chain. |
| VIII. One parameter space | PASS | Only the label changes; parameters fixed by one declared rule (R2). |
| IX. Orthogonality | PASS (measured) | Label–width Spearman recorded for every candidate; S1 measures separation within the width book, i.e. beyond width. |
| X. Structural fixes | PASS | No pair-specific code. |
| XI. No leakage | PASS | Feature lists identical to the control (R10); thresholds from training data only; late half unread for non-winners. |
| Framework strategy-agnostic | PASS | No Framework/ change. |
| Params merge | PASS | Classes inherit the control; `.json` copied from the control; LABEL_PARAMS merges over base params. |
| Quality gates | PASS | Pre-registration committed before training; judge refuses stale records; small-margin seed-2 rule. |

**Post-design re-check**: no violations. One finding from planning is folded into the design rather
than the spec's scope: mfe_mae_ratio has no sell variant (R1); the S0a audit decides whether it is
trainable, without a Framework change.

## Project Structure

### Documentation (this feature)

```text
specs/034-label-type-screen/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/candidate_classes.md, contracts/screen_cli.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code (nested repository `user_data/strategies`)

```text
NNNC/
├── NNNC_LS_<TAG>.py / .json        # generated: label candidate (entry-source role)
└── NNNC_LS_<TAG>_SZ.py / .json     # generated: sizer twin (same model key)
research/label_screen/
├── candidates.py        # the declared candidate table (R2), single source for every script
├── build_arms.py        # generate classes and plans (sz / es) from the control plan
├── audit.py             # S0a: label shares, overlap, param liveness, feature-list identity
├── train.py             # 032 driver, registered per candidate, early folds by default
├── pick.py              # sizer 0.5-share picks (033 pick, generalised) + entry-rate picks (pick_threshold)
├── separation.py        # S1: gap and AUC per candidate vs both references
├── survivors.py         # applies the survivor rule, writes survivors.json
├── judge.py             # staged verdicts S0..S3 (033 judge pattern, two roles)
├── preregistration.json / .md
├── audit.json, instrument.json, separation.json, survivors.json, verdicts.json, report.md
└── tests/
research/nnnc_framework/signals.py  # + LS_PLANS registration (as CO_/WR_PLANS)
research/nnnc_framework/diagnose.py # + runtime MODELS registration from the screen scripts
```

**Structure Decision**: all study code in `research/label_screen/`, generated strategy classes in
`NNNC/`, and two one-line registrations in shared drivers (the 032/033 pattern). 033's pick,
separation and judge are generalised by import and parameter, not copied wholesale.

## Complexity Tracking

No violations to justify.
