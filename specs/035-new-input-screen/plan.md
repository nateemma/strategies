# Implementation Plan: New-Input Screen for NNNC

**Branch**: `035-new-input-screen` (spec directory; committed to master) | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/035-new-input-screen/spec.md`

## Summary

Build 17 inputs the model has never seen, causally aligned per pair and 15-minute bar. They come
in four families: order flow (global and Binance.US), perp positioning, market state, and spot–perp
basis. Then ask of each: does it rank the 412 early width-book trades beyond bb_width and the
current model's p_buy? The stages:

1. **S0:** download and verify the free archives; build a per-pair input cache; prove causality by
   truncation; record coverage and live feasibility.
2. **S1** (no training): the partial Spearman correlation with trade pp given bb_width and p_buy,
   plus an orthogonality cap and sign stability. At most 3 survivors.
3. **S2:** train ARO with each survivor added on its own (early folds) and score the 0.5-share
   separation gap against the control's 0.1004.
4. **S3:** the winner trains on the late folds and is backtested as the control sizer with only its
   weights changed, against `cs_aro` and `w_aro` (judge_one), with `wf_aro_g24` reported.

**Technical approach:**
- No Framework/ change.
- Inputs are normalised causally in the builder and declared pre-normalised, so the control's
  scaler and every other run input stay identical (R1, R2).
- A default-off `NewInputMixin` in `NNNC/` merges the cache.
- Study code lives in `research/new_inputs/`, reusing 034's separation, the control-sizer picker
  and instrument, and the 032 training driver.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**: freqtrade, MLX (LSTM_RAW), numpy, pandas, scipy, talib (BTC ADX), stdlib
`urllib`/`zipfile`/`hashlib` for the archives

**Storage**:
- raw archives and manifest in `user_data/data/external/`, and the input cache in
  `user_data/data/new_inputs/` (both outside the strategies repo, uncommitted);
- weights at new model keys;
- plans in `evaluation/folds/plan_ni_*`;
- signal tables `research/nnnc_framework/signals/NI_<TAG>_fNN.parquet` (git-ignored);
- records in `regime/arms/`;
- study outputs in `research/new_inputs/`.

**Testing**: pytest in a new `research/new_inputs/tests/`. The full suite (1064 passed, 9 skipped)
stays green.

**Target Platform**: macOS on Apple Silicon, local; outbound HTTPS to data.binance.vision and
api.binance.us.

**Project Type**: research framework (nested strategies repository).

**Performance Goals**:
- downloads about 30–45 min once (about 17,400 metrics files, about 800 kline files, and about
  2,000 Binance.US REST pages);
- build in minutes; the truncation test is 200 shared-cut all-pair rebuilds (R12), estimated at
  up to about an hour;
- S1 in seconds;
- S2 at most 3 × 8 ARO folds (about 3 min each), plus tables and 24 picks;
- S3 7 folds of training, 15 picks and 15 backtests;
- conditional: seed-2 (it needs ARO seed-2 weights on 15 folds for `cs_aro`).

**Constraints**: one chain at a time; picks one fold per process; byte-identical when off;
Framework/ strategy-agnostic; downloads treated as untrusted data; live bot untouched; nothing
promoted.

**Scale/Scope**: 17 inputs, 10 pairs, ARO only.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / constraint | Status | How |
|---|---|---|
| I. Deployment regime | PASS | 2023–26 walk-forward. The late half is read only for the winner. |
| II. Distant eras | PASS by restriction | No distant era with these sources for NNNC: at most PASS (PROVISIONAL), with a forward route. |
| III. Learnability ≠ edge | PASS | S1 is scored on realised trade pp. The verdict needs S3 backtest P&L against the current sizer. |
| IV. Parameters live | PASS | Mixin byte-identity test when off; the survivor key must differ; S2 checks the new column is present and non-constant in the training frame. |
| V. Lookahead / realism | PASS | A truncation test of every input at 200 shared cuts (R12), plus an end-to-end check through the strategy merge (T011), conservative 5-minute alignment (R3), and the same execution config and liquidity cap at S3. |
| VI. Diagnose before tuning | PASS | Coverage and orthogonality are recorded before any training. Nothing is tuned. |
| VII. Validate cheaply | PASS | S1 needs no training and gates S2; S2 gates S3. |
| VIII. One parameter space | PASS | One input per survivor model; everything else equals the control (R1, R2). |
| IX. Orthogonality | PASS | |Spearman with bb_width| ≤ 0.5 is a survival condition, and correlation with p_buy is recorded. |
| X. Structural | PASS | Identical definitions for every pair; no per-pair code. |
| XI. No leakage | PASS, with a recorded exposure | New inputs have no fitted artefact (rolling, causal). The control's shared main scaler (fit range unrecorded) is read identically by control and survivors; recorded in the report, and fixing it is out of scope (R1). |
| Framework strategy-agnostic | PASS | Mixin in `NNNC/`; no Framework/ change. |
| Params merge | PASS | Classes inherit the control; `.json` copied from the control. |
| Quality gates | PASS | Pre-registration committed before S1; the judge refuses stale records; small-margin seed-2 rule. |

**Post-design re-check:** no violations. One spec change came out of planning. FR-013 is amended
(R1): the new inputs carry no fitted scaler, and the original columns use the control's scaler
unchanged. Per-fold scalers for the survivor alone would have been a second variable.

## Project Structure

### Documentation (this feature)

```text
specs/035-new-input-screen/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/screen_cli.md, contracts/new_input_mixin.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code (nested repository `user_data/strategies`)

```text
NNNC/
├── _new_inputs.py                  # NewInputMixin (default off)
├── NNNC_NI_<TAG>.py / .json        # generated per survivor
└── NNNC_NI_<TAG>_SZ.py / .json     # generated sizer twin (same key)
research/new_inputs/
├── inputs.py          # the declared input table (R6), single source for every script
├── fetch.py           # archive and REST downloader: manifest, checksums, resumable
├── sources.py         # archive parsers (µs/ms detection, headers), as-of alignment (R3, R4)
├── build.py           # per-pair input cache + build_meta.json
├── causality.py       # R12 truncation test + planted control
├── coverage.py        # S0 coverage + live-feasibility record
├── screen.py          # S1 (reproduction gate, partial Spearman, split gap)
├── survivors.py       # FR-010
├── build_arms.py      # classes and plans for survivors
├── train.py           # 032 driver registration (as 034)
├── separation.py      # S2 gap vs 0.1004 (034 separation, NI tables)
├── instrument.py, judge.py
├── preregistration.json / .md
├── causality.json, coverage.json, screen.json, survivors.json, separation.json,
│   instrument.json, verdicts.json, report.md
└── tests/
research/control_sizer/pick.py        # + optional --plan (default unchanged)
research/nnnc_framework/signals.py    # + runtime NI_ registration (as LS_)
```

**Structure Decision:** study code in `research/new_inputs/`, one mixin plus generated classes in
`NNNC/`, data outside the repository, and small default-preserving extensions to two shared tools.

## Complexity Tracking

No violations to justify.
