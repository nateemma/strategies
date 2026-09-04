# Implementation Plan: Tail-Targeted Breakout Labelling

**Feature Directory**: `specs/003-breakout-tail-labelling` (commits go directly to master) | **Date**: 2026-09-04 | **Spec**: [spec.md](./spec.md)

## Summary

The breakout label rewards moves that mostly cannot pay for the stop: median
labelled gain 2.83% against a −4% to −6% operative stop, only 26% of labelled bars
ever clearing it, reward:risk 0.57. Seven prior interventions all acted downstream
of the label; this is the one structural cause untested.

Phase 0 established what is and is not available. The economics CAN be fixed —
`H=96/0.03` reaches reward:risk 1.28 with 67% of labels clearing the stop — but
density is the binding constraint (two of five candidate cells fall below the
all-Hold floor), and the best cell's payoff peaks at 17h against a hardcoded 12h
forced exit. So the headline candidate is a **package**, and `H=48/0.02` exists as
the cell that isolates the label effect with no exit change at all.

Two findings shape the method more than the candidates do. **Six of six
model-side changes across two families produced byte-identical out-of-sample
trades** — so "identical trades" is the expected outcome and must be reported as
inconclusive, not as refutation. And **three separate metrics have each failed to
predict return here**, one by 43pp in the wrong direction — so acceptance is
realised out-of-sample return alone.

Approach: screen by calculation (minutes, no training), admit or exclude on
density and hold-window conflict, then train survivors sequentially cheapest-first,
isolating the label before testing the package.

## Technical Context

**Language/Version**: Python 3.12 under `.venv/bin/python`, `PYTHONPATH=user_data/strategies`.

**Primary Dependencies**: pandas, numpy, scipy (`stats.skew`/`kurtosis`), freqtrade for backtests and `freqtrade.data.btanalysis` for recorded trades, MLX for training. No new third-party dependency.

**Storage**: Price history `user_data/data/binanceus/*-15m.feather`. Arm records `regime/arms/<tag>_<window>.json`. Trained weights `saved_data/<Strategy>/`. Outputs are markdown plus the existing arm-record format.

**Testing**: Correctness is established by reproduction, not unit tests — the re-baselined control must reproduce the parked control's trade hashes exactly (R8), and every arm is checked for trade-difference against that control before its numbers are read.

**Target Platform**: macOS, Apple Silicon, 14 cores, local execution.

**Project Type**: Research study on one host strategy, plus a per-strategy exit override if the package candidate is reached.

**Performance Goals**: Screening completes in minutes per candidate. Each retrain is 8–45 min depending on label density. Backtests are ~1 min per window.

**Constraints**: Pinned literal timeranges. Training ends before the out-of-sample window begins. Realistic execution constraints active. Nothing retrains on gbb/NNNC. **Sequential execution** — two concurrent jobs earlier produced load 60 on 14 cores and ran slower than serialising.

**Scale/Scope**: One host, 5 screened candidates of which 3 survive density, 11 traded pairs, 3 pinned windows of which 1 is out-of-sample.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1. Constitution v1.0.0.*

| Principle | Verdict | How this plan satisfies it |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 carries the verdict and is the only window outside training. W2/W3 are reported as in-sample and excluded from acceptance (FR-013). |
| II. Persistence across distant eras | **PASS with a caveat stated** | W3 cannot serve its usual role here — it lies inside the training window. The plan does not claim persistence evidence it cannot have; W3 is reported as fit quality. This is a deliberate trade to keep W1 genuinely out-of-sample, recorded in Assumptions. |
| III. Learnability is not edge | **PASS — the feature's spine** | FR-014 forbids metric-based acceptance outright. R7 documents three metrics each failing, one by 43pp in the wrong direction. |
| IV. Verify parameters are actually live | **PASS — mandatory gate** | FR-015 makes the trade-difference check a precondition for reading any delta. R6 shows six of six prior changes were absorbed. R8 fixes a live params-file confound worth 12pp. |
| V. Lookahead-clean is not realistic | **PASS** | Volume/liquidity constraints stay active. Screening reads only historical price paths; no lookahead surface. |
| VI. Diagnose before tuning | **PASS — exemplary** | The entire candidate set derives from measurement (R1–R3), and R2 identifies the defect before any tuning is proposed. |
| VII. Validate cheaply before building | **PASS** | Phase 1 is calculation-only and excludes two of five candidates before a retrain is spent. |
| VIII. One parameter space at a time | **PASS** | Label first in isolation (FR-004), then the package with contributions separated (FR-011), risk side only if a label is adopted. R8 explicitly refuses to bundle a take-profit change into a labelling feature. |
| IX. Orthogonality earns a factor its place | **PASS (N/A)** | No multi-factor book. |
| X. Structural fixes, not per-pair | **PASS** | FR-021. The unclog change is per-STRATEGY, never per-pair. |

**Post-Phase-1 re-evaluation**: unchanged. The only code touched outside the host
strategy is a new labeler variant in `TrainingSignals.py` (additive, registered
under a new method id) and a per-strategy method override. No `BaseStrategy`
default changes, so Principle X and the framework-promotion rule are not engaged.

**Result: no violations.** Complexity Tracking is empty and has been removed.

## Project Structure

### Documentation

```text
specs/003-breakout-tail-labelling/
├── plan.md              # This file
├── spec.md              # 22 FRs, 8 SCs, 5 user stories
├── research.md          # Phase 0 — R1..R9, all resolved
├── data-model.md        # Phase 1 — entities and validation rules
├── quickstart.md        # Phase 1 — runnable validation guide
├── contracts/
│   └── study-cli.md     # Script invocations and record schemas
├── checklists/
│   └── requirements.md  # 16/16 pass
└── tasks.md             # Phase 2 — created by /speckit-tasks, NOT here
```

### Source Code (repository root = `user_data/strategies/`)

```text
regime/
├── breakout_dist.py            # EXISTS — distribution shape
├── breakout_tail.py            # EXISTS — economics by (H, threshold)
├── tail_path.py                # EXISTS — reachability and time-to-peak
├── label_path_audit.py         # EXISTS — drawdown before the labelled profit
├── screen_candidates.py        # NEW — one record per candidate (FR-001)
└── run_arm.py                  # EXISTS — unchanged

NNNC/
├── NNNC_BreakoutOpen.py        # host — unchanged
├── NNNC_BreakoutOpen.json      # NEW — pins the effective config (R8, FR-016)
├── NNNC_Brk_L<n>.py + .json    # NEW per surviving candidate, deleted after
└── ...

Framework/
└── TrainingSignals.py          # MODIFIED only if the path-aware variant is reached
```

**Structure Decision**: the study lives under `regime/` beside features 001 and
002 because it reuses `run_arm.py`, the pinned windows and the arm-record format
directly. Candidate arms are throwaway subclasses in `NNNC/`, each with its own
params file, deleted after measurement. `Framework/` is touched only for an
additive labeler; the unclog change is a per-strategy override on the host, never
an edit to `BaseStrategy` (R4).

## Execution Phases

### Phase 1 — Screen (no training)

Produce one screening record per candidate: density, median labelled gain,
stop-clearing share, reward:risk, time-to-peak, stop-before-peak share
(FR-001). Exclude any candidate below the density floor (FR-002). Nothing loads
or trains a model (FR-003).

Most figures already exist from R1–R5. What remains: the path-aware variant and
the alternative-basis label have not been screened at tail settings.

**Exit condition**: every candidate admitted or excluded on recorded numbers,
with the hold-window conflict flagged per candidate.

### Phase 2 — Re-baseline the control

Write `NNNC/NNNC_BreakoutOpen.json` pinning the currently effective configuration
and re-run the three control windows. **The new control must reproduce `brkfix`'s
trade hashes exactly**; if it does not, the params file changed behaviour and that
must be resolved before any arm is compared (R8).

**Exit condition**: a control whose configuration is explicit and whose hashes
match the parked run. 3 backtests, no retrain.

### Phase 3 — Isolate the label

Train and evaluate the candidate that requires **no exit change** (FR-004), so
the label effect is measured without the hold window confounding it.

Per arm: verify `exit_config_hash` matches the control; compare `trades_hash`;
report W1 as the verdict with W2/W3 marked in-sample; report trade counts.

**Exit condition**: a classification of improved / unchanged / worsened /
inconclusive-because-absorbed, with the trade-difference check recorded (SC-003).

### Phase 4 — The package

Only if Phase 3 is not conclusively negative. Add the per-strategy hold-window
override and the tail label, measuring the hold-window change **both with and
without** the label change so the two contributions separate (FR-011).

**Exit condition**: separated contributions, and no shared framework file left
modified (FR-010, SC-006).

### Phase 5 — Risk side (contingent, P3)

Only if a label is adopted. Re-sweep the ATR band and entry threshold **against
that label's own control**, not the historical one. Not elaborated further here.

## Per-Arm Protocol

| Gate | Mechanism |
|---|---|
| Pinned windows | W1 `20250901-20260831` (OOS, verdict), W2 `20240601-20250531`, W3 `20230101-20231231` (both in training) |
| Training window | `20220101-20250831`, ends before W1 begins |
| Configuration | Arm carries its own params file; `exit_config_hash` compared to control |
| Liveness | `trades_hash` vs control. Identical ⇒ INCONCLUSIVE, never refuted |
| Judge on | Realised W1 return. Never a classifier metric |
| In-sample | W2/W3 reported and labelled, excluded from acceptance |
| Power | Trade counts reported; sub-1pp W1 deltas treated as indistinguishable (R9) |
| Execution | Sequential. Realistic execution constraints active |

## Known Hazards

1. **The gate absorbs the change.** Six of six prior model-side changes produced
   identical out-of-sample trades. This is the base rate, not the tail risk.
   Mitigated by FR-015 making it a distinct reported outcome.
2. **`unclog_12h` is hardcoded** at `Framework/BaseStrategy.py:1040` and a live
   dry run imports that file. Mitigated by using a per-strategy override rather
   than an edit-revert cycle.
3. **A params file silently changes the exit config.** Cost 12pp on the GAN
   comparison. Mitigated by Phase 2's hash-reproduction requirement.
4. **Low power.** 97 control trades on W1; sub-1pp deltas are noise (R9).
5. **In-sample flattery.** Two of three windows are inside training; six prior
   interventions produced positive summed returns that were entirely in-sample.
6. **`test_strat.sh -f` deletes before retraining**, and `TaskStop` does not kill
   the process tree — kill by pattern. Both cost real time today; one destroyed a
   pre-existing model that had no backup.
7. **Density collapse.** Raising the threshold reduces labels; below ~5.5% of bars
   the classifier stops predicting the signal class entirely.
8. **A negative result is the likely outcome** and is a deliverable (FR-022). The
   plan must not drift toward manufacturing a positive from in-sample windows.

## Next Command

`/speckit-tasks` — to generate the dependency-ordered task breakdown.
