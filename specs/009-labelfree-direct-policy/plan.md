# Implementation Plan: Label-free Direct Policy Optimisation

**Branch**: `009-labelfree-direct-policy` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/009-labelfree-direct-policy/spec.md`

## Summary

Test whether training a small long-only exposure policy directly on net-of-cost realised return
converts more of the available signal into P&L than the current label-trained classifier. The
work is strictly ordered: build a vectorised differentiable simulator and prove it reproduces the
arbiter's P&L on a fixed, already-realised trade series; prove the learning path can exploit
deliberate lookahead; only then train the policy across a 12-configuration grid at ≥5 seeds,
screened on one fold before a pooled 15-fold run; finally judge it against four mandatory
baselines with cost stress, overfitting statistics and a byte-identical reproduction.

Technical approach: MLX on Apple Silicon, `mlx.nn.LSTM(64)` over the existing `seq_len = 16`
feature window, two objectives (minibatch-unbiased mean-variance utility; robust worst-window
Sharpe) × two turnover controls (learned no-trade band; explicit turnover penalty) × three cost
scales. Costs, an OHLCV-estimated half-spread and a soft-min liquidity cap sit inside the training
objective. The freqtrade backtester remains the arbiter; the simulator exists only for training
speed.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)

**Primary Dependencies**: MLX (`mlx.nn.LSTM/GRU/Linear`, `mlx.core`) for the policy and simulator;
pandas/numpy for data assembly; existing repo modules — `Framework/FeatureNormalizer.py` (scalers),
`utils/DataframePopulator.py` (indicators), `regime/wf/` (fold plan, pooling, comparison),
`regime/regimedata.py` (pinned windows). PBO via CSCV and the Deflated Sharpe Ratio for trial
accounting.

**Storage**: Feather OHLCV under `user_data/data/binanceus/` (columns
`date, open, high, low, close, volume` only — no bid/ask); trained policies and scalers under
`saved_data/`; append-only trial log at `research/labelfree/trials.jsonl`; backtest archive zips
under `user_data/backtest_results/`.

**Testing**: Calibration test against the arbiter (blocking); deliberate-lookahead sanity ceiling;
cross-process determinism check; the repo's existing pooled walk-forward harness as the evaluation
instrument.

**Target Platform**: macOS on Apple Silicon (Metal GPU), single machine, sequential execution only.

**Project Type**: Research feature — offline training and evaluation code plus one thin deployable
strategy wrapper. No service, no UI.

**Performance Goals**: Policy training in minutes per fold (the model is tiny). The simulator must
be fast enough that a 12-config × 5-seed grid on one fold is a single sitting; the pooled 15-fold
run for two configs is the long job. For scale, the existing control's full 15-fold run took
~113 minutes sequentially.

**Constraints**: Decision at bar `t`, fill at `t+1`, no other lookahead. Costs, half-spread and the
10%-of-volume fill cap inside the objective. Sequential runs only. Outer freqtrade checkout receives
no commits; only files this feature creates are staged.

**Scale/Scope**: 10-pair walk-forward universe, 15-minute bars, 15 anchored folds (730-day minimum
training, 90-day test), 378 pooled control trades as the comparison base. Grid: 12 configurations ×
≥5 seeds on one screening fold, then the best two over all 15 folds.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Gate | Pre-Phase-0 | Post-Phase-1 |
|---|---|---|---|
| I. Evaluate in the deployment regime | W1 is the headline; W2/W3 are context | PASS — spec Assumptions | PASS |
| II. Persistence across distant eras (NON-NEGOTIABLE) | A distant era must be tested | PASS — SC-004 requires W3 | **PASS with recorded risk** — see Complexity Tracking |
| III. Learnability is not edge | No model-fit metric may decide acceptance | PASS — every SC is cost-aware P&L; no MCC/R² anywhere | PASS |
| IV. Verify parameters are actually live | Changed behaviour must be measured downstream | PASS — SC-002 sanity ceiling; FR-014b reconciliation | PASS |
| V. Lookahead-clean is not realistic | Realistic execution mandatory | PASS — FR-004/004a/005 put fees, estimated spread and the volume cap inside the objective | PASS |
| VI. Diagnose before tuning | Diagnostics before hyperparameters | PASS — calibration localises fee/fill/stake disagreement (research R11) before any sweep | PASS |
| VII. Validate cheaply before building | Cheap gate before expensive build | PASS — calibration and sanity ceiling gate the grid | PASS |
| VIII. One parameter space at a time | No joint search on thin data | ⚠️ **JUSTIFIED EXCEPTION** — a 12-cell factorial is a joint search; see Complexity Tracking | ⚠️ JUSTIFIED EXCEPTION |
| IX. Orthogonality | Factor correlation checked first | N/A — single policy, not a multi-factor book | N/A |
| X. Structural fixes, not per-pair overrides | No pair-specific code | PASS — FR-002 pair-agnostic features; Non-Goals bars per-pair code | PASS |
| Repository boundary | Outer checkout untouched | PASS — FR-031 | PASS |
| Framework stays strategy-agnostic | No strategy code in `Framework/` | PASS — research R12 places code in `research/labelfree/` and `NNDP/` | PASS |
| Parameter dicts merge from parent | No fresh literal dicts | PASS — wrapper must merge, see contracts | PASS |
| Workflow gate 1 — pinned time range | Ranges pinned | PASS — `regime/regimedata.py` windows, fold plan |PASS |
| Workflow gate 2 — distant-era window | ≥3 windows, one distant | PASS with risk (see below) | PASS with risk |
| Workflow gate 3 — realistic execution | Volume/liquidity active | PASS — FR-005 | PASS |
| Workflow gate 4 — liveness proof | Downstream measurement | PASS — SC-002, SC-018 | PASS |
| Workflow gate 5 — paired seed robustness | Sub-1pp deltas need paired runs | PASS — FR-016 ≥5 seeds, IQM | PASS |
| Workflow gate 6 — trade count reported | Zero-trade arm is inconclusive | PASS — FR-022, Edge Cases | PASS |
| Workflow gate 7 — scalers regenerated | Regenerate on include-list change | PASS — Assumptions: include list unchanged | PASS |

**Result: no unjustified violations.** One risk is recorded in Complexity Tracking rather than
treated as a violation, because the gate is still met — the distant window *is* tested; what is
degraded is its interpretability.

## Project Structure

### Documentation (this feature)

```text
specs/009-labelfree-direct-policy/
├── plan.md              # This file
├── spec.md              # Feature specification (39 FR, 18 SC)
├── research.md          # Phase 0 output — 12 decisions with measurements
├── data-model.md        # Phase 1 output — entities and validation rules
├── quickstart.md        # Phase 1 output — runnable validation guide
├── contracts/           # Phase 1 output — module and file-format contracts
│   ├── simulator.md
│   ├── trial-log.md
│   └── strategy-wrapper.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (16/16)
└── tasks.md             # Phase 2 — created by /speckit-tasks, NOT by this command
```

### Source Code (repository root = `user_data/strategies/`)

```text
research/labelfree/                 # NEW — all research code for this feature
├── __init__.py
├── preflight.py                    # Prerequisite check: data, scalers, control record, config
├── folds.py                        # Windows + anchored fold plan + per-fold pair universe
├── archive.py                      # Reads the arbiter's realised trade series from the archive
├── sim.py                          # Vectorised differentiable simulator (protocol §1)
├── costs.py                        # Fee, Abdi-Ranaldo half-spread, vol-scaled slippage
├── policy.py                       # LSTM(64) backbone + exposure head + no-trade band
├── losses.py                       # Mean-variance utility; robust worst-window Sharpe
├── train.py                        # Per-fold training loop, seeds, truncated BPTT
├── sweep.py                        # Variant-grid driver; writes trial records before each run
├── trials.py                       # Append-only trial log writer
├── audit_trials.py                 # Trial-log audit (final gate)
├── calibrate.py                    # Blocking calibration against the arbiter
├── reconcile.py                    # SC-018 simulator-vs-arbiter P&L and exit-source share
├── baselines.py                    # Buy-and-hold; pinned 20/60 EMA trend rule with band
├── stress.py                       # Cost stress: fee multiples and tighter liquidity cap
├── export_policy.py                # Writes policy weights; pins the wrapper's model path
├── reproduce.py                    # Fresh-process reproduction from a saved config + seed
├── stats.py                        # IQM, bootstrap CIs, turnover/hold time, DSR, PBO (CSCV)
├── trials.jsonl                    # Append-only trial log (created at first run)
└── tests/
    ├── test_sim_calibration.py
    ├── test_sim_contract.py
    ├── test_lookahead_sanity.py
    └── test_determinism.py

NNDP/                               # NEW — deployable wrapper family
└── NNDP_MLX.py                     # BaseNNStrategy subclass; get_predictions loads the policy

research/label-free-study/briefs/results/   # NEW — study reports
├── 01-calibration-2026-09-10.md    # Blocking calibration report
└── 01-<date>.md                    # Final result report (protocol §6)
```

Existing directories consumed read-only: `Framework/`, `utils/`, `MLX/`, `regime/`, `config/`,
`saved_data/`. **Nothing is added to `Framework/`** (constitution).

**Structure Decision**: Research code is isolated under `research/labelfree/` because it is
consumed by this feature and four sibling briefs, and must not contaminate the strategy-agnostic
`Framework/`. The only production-shaped artefact is `NNDP/NNDP_MLX.py`, a thin wrapper that lets
the arbiter score the trained policy as a real strategy. This mirrors the existing split between
`regime/` (research instruments) and the strategy families.

## Complexity Tracking

> Recorded risks that are *not* constitution violations but must not be discovered late.

| Item | Why it is accepted | Simpler alternative rejected because |
|------|--------------------|--------------------------------------|
| W3 trains on 9 of 10 pairs; six of them on partial history (345-533d vs 729d). Only XRP is excluded, for a 911-day delisting gap | Principle II satisfied — a distant era *is* tested, ZEC **is** present, and the control faces the identical universe so the comparison stays paired. **Corrected 2026-09-10 during implementation**: the earlier "4 of 10, ZEC excluded" figure came from a start-date test, which wrongly counts XRP as covered | Shortening W3 training confounds with training length; restricting every window to the fully-covered pairs breaks comparability with the control's pooled record (both rejected at Q5) |
| Training objective omits the deployment exit stack | Keeps the wrapper comparable to the control (FR-028) and avoids building exit machinery no downstream brief consumes | Replicating take-profit/stop/grace inside the objective reintroduces exactly the scope Q1 removed; instead FR-014b/SC-018 force the resulting gap to be measured |
| Two objectives × two turnover controls × three cost scales = 12 configs before any pooled run | The literature's central finding is that the *naive* direct-P&L version overfits and only structured versions hold, so running one variant cannot answer the question | A single configuration would make a null result uninterpretable; the single-fold screen keeps the pooled cost to two configs |
| **Principle VIII exception: a 12-cell factorial is a joint search over three dimensions** | The constitution permits a violating proposal to carry "an explicit, written justification naming the principle and the reason the exception is warranted". Naming it: **Principle VIII, One Parameter Space at a Time.** The reason: the literature's central finding is that the *naive* direct-P&L objective overfits (val 1.61 → test 0.48 Sharpe) and only the structured version holds, so the **interaction** between objective and turnover control is the object of study, not a nuisance to be held fixed. Tuning one dimension while the other is mis-set would measure the wrong thing and could kill the family on a configuration nobody proposed. Mitigations are mandatory, not optional: every cell is pre-registered in the trial log, the full trial count feeds DSR and PBO, and PBO > 0.2 fails the feature outright | Sequencing (lock objective → lock turnover control → sweep γ) reduces trials from 12 to 7 and would be the default under Principle VIII, but it presumes the dimensions are separable — which is exactly the claim the brief exists to test. Rejected on those grounds, with the exception documented here rather than taken silently |
| **Screening on a fold with 71 trades sits inside the measured noise floor** | AGENT_GUIDE's measurement is that a ~100-trade window carries a ±7.6pp noise floor and that "the optima are not trackable" — a whole parameter sweep fits inside that band. The screening fold (f08) has 71 control trades, so a rank ordering of 12 configurations on it is *presumptively* a ranking of noise. Accepted only because the alternative — screening on the pooled 15-fold run — costs ~113 min per arm × 12 configs × 5 seeds, which exceeds the brief's entire budget | Mitigated rather than ignored: the selection task must bootstrap the between-configuration deltas and state whether the top-2 choice is resolvable. When it is not, configurations are carried forward by a **pre-registered rule** rather than by rank, and the result report must say the screen was unresolved |
| User stories are strictly sequential, not independently deliverable in parallel | Each story is a gate on the *credibility* of the next, not merely on its inputs: an unproven gradient path (US2) makes a null sweep (US3) uninterpretable, and an uncalibrated simulator (US1) makes every downstream number fiction. Protocol §5 states this as a kill condition, not a preference | Forcing artificial independence would let the sweep run before the leakage ceiling passes, which is exactly the failure the protocol's ordering exists to prevent. US1 remains independently valuable — it is the shared dependency of briefs 2, 4, 5 and 6 |
| Byte-identical reproduction required on GPU | Protocol §4 demands it and a seeded MLX computation reproduced exactly across processes in testing | Accepting a numerical tolerance would weaken a protocol gate, which the study forbids |
