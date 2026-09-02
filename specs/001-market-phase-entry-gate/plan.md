# Implementation Plan: Market-Phase Framework and Phase-Adaptive Entry Gate

**Branch**: `001-market-phase-entry-gate` (spec directory; commits go directly to master) | **Date**: 2026-09-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-market-phase-entry-gate/spec.md`

## Summary

Build a three-level market-phase classifier (asset, market breadth, BTC), characterise it until only instruments with actionable persistence remain, then answer one question: does a phase label predict which of two entry aggressiveness settings wins the coming period, better than always using one fixed setting? Only on a PASS do the phase-adaptive entry mechanisms get built.

The prior BTC gate failed because a market-wide taxonomy scored on *passive* returns was applied as a *hard binary filter* to *active buy-low* models. This plan changes all three: per-asset and breadth levels alongside BTC, scoring on the host's own realized trades, and soft modulation of the dominant entry filter rather than an AND-ed boolean.

Approach: a standalone study directory (`regime/`) carries Phases A–C with no strategy changes at all. Only Phase D touches strategy code, and only if Phase C passes.

## Technical Context

**Language/Version**: Python 3.12, run under `.venv/bin/python` with `PYTHONPATH=.` from the freqtrade root — pandas is not on the system interpreter.

**Primary Dependencies**: pandas, numpy, talib (already used by `btc_corr/`), matplotlib (Agg backend), rapidjson. freqtrade for the backtest arms. No new third-party dependency is introduced.

**Storage**: Feather OHLCV under `user_data/data/binanceus/` (78 pairs at 15m; BTC spans 2021-01-01 to 2026-08-31, 198,486 bars). Study outputs are CSV plus PNG under `regime/`, matching the `btc_corr/` convention. Backtest arms export trades to per-arm files.

**Testing**: End-to-end lookahead validation (truncated versus full history label identity) is the primary correctness test for Phases A–B. Phase C and D are validated by measurement, not unit tests. Existing pytest suites under `Framework/tests/` are only touched if Phase D changes `Framework/`.

**Target Platform**: macOS on Apple Silicon, local execution.

**Project Type**: Research study plus an opt-in strategy mixin. Not a service or application.

**Performance Goals**: A full classifier pass over 78 pairs × 198K bars completes in a single run without chunking. Phase A/B are re-runnable in minutes so the characterisation can be iterated; Phase C/D backtest arms are the expensive part and are budgeted per arm, not per run.

**Constraints**: Every instrument must be causal and pass end-to-end lookahead validation. Every comparison runs on pinned literal date ranges under realistic execution constraints. No modification to exit behaviour. No per-pair special-casing.

**Scale/Scope**: 78 pairs, three phase levels, roughly a dozen candidate instruments, three evaluation windows, and — only on a PASS — four mechanism arms plus a baseline.

## Constitution Check

*GATE: evaluated before Phase 0 research, re-evaluated after Phase 1 design. Constitution v1.0.0.*

| Principle | Verdict | How this plan satisfies it |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | W1 (2025-09→2026-08) is the deployment regime and carries the headline verdict per FR-027a. W2/W3 are context and persistence checks. |
| II. Persistence across temporally distant eras | **PASS** | W3 (2023) is roughly two years from W1. Research R3 explicitly records that W1 and W2 are only three months apart and cannot establish persistence between them. |
| III. Learnability is not edge | **PASS** | No model-fit metric appears in any acceptance rule. FR-029 and the User Story 3/4 scenarios decide on P&L and risk-adjusted return only. |
| IV. Verify parameters are actually live | **PASS** | R7 defines a three-count liveness measurement per arm. R5 records the specific live hazard here: `NNNC_MLX.json` overrides `buy_params`, so a Python-side threshold change would silently do nothing. |
| V. Lookahead-clean is not realistic | **PASS** | R9 specifies truncated-versus-full-history validation at three cut points, explicitly rejecting freqtrade's own lookahead analysis as sufficient. FR-028 keeps volume constraints active on every arm. |
| VI. Diagnose before tuning | **PASS** | Phase B characterisation and the Phase C verdict both precede any parameter change. No mechanism is touched before the diagnostic phases complete. |
| VII. Validate cheaply; trackability before adaptivity | **PASS** | Phase C *is* this gate, and it reuses the harness shape of the tool that previously killed a dynamic lookback. Phase D is explicitly unreachable on a FAIL. |
| VIII. One parameter space at a time | **PASS (N/A)** | No hyperparameter search is in scope. Mechanisms are discrete arms measured one at a time, in a fixed order. |
| IX. Orthogonality earns a factor its place | **DEFERRED** | Applies only to mechanism M4, where phase features would join an existing model input set. Recorded as a gate on M4 below; not applicable to Phases A–C. |
| X. Structural fixes, not per-pair overrides | **PASS** | FR-025 and the Out of Scope section. No instrument or mechanism is permitted to reference a specific pair. |

**Post-Phase-1 re-evaluation**: unchanged. The Phase 1 design introduces no new violation — the study directory is standalone, the strategy-side mixin is opt-in and defaults off, and no `Framework/` file is modified in Phases A–C.

**Result: no violations requiring justification.** The Complexity Tracking table is therefore empty and has been removed.

## Project Structure

### Documentation (this feature)

```text
specs/001-market-phase-entry-gate/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output — R1..R11, all unknowns resolved
├── data-model.md        # Phase 1 output — entities and their validation rules
├── quickstart.md        # Phase 1 output — runnable validation guide
├── contracts/
│   ├── study-cli.md     # Study script invocations and exit semantics
│   └── artifacts.md     # Output file schemas (instrument catalogue, dwell, verdict, arms)
├── checklists/
│   └── requirements.md  # Spec quality checklist (all items pass)
└── tasks.md             # Phase 2 — created by /speckit-tasks, NOT by this command
```

### Source Code (repository root = `user_data/strategies/`)

```text
regime/                              # NEW — the study, mirrors btc_corr/ conventions
├── regimedata.py                    # panel loading, universe assembly, coverage rule (R4)
├── instruments.py                   # instrument definitions incl. exp4 baseline (FR-004)
├── phase_a_classify.py              # emit phase labels at all three levels (FR-001..FR-005)
├── phase_a_lookahead.py             # truncated vs full history validation (FR-007, R9)
├── phase_b_characterize.py          # distribution, dwell, transitions, per era (FR-008..FR-011)
├── phase_b_risk_normalize.py        # time-normalized drawdown re-analysis (FR-012)
├── phase_c_trackability.py          # THE VERDICT (FR-013..FR-018)
├── run_arm.py                       # backtest runner: pinned range, trade export, pass rates (R7, R8)
└── README.md                        # findings, including a FAIL verdict if that is the outcome

utils/
└── custom_indicators.py             # MODIFIED — add variance_ratio, hurst_from_vr, rolling_autocorr (R1, R2)

# Phase D only, and only on a PASS verdict:
utils/
└── PhaseGate.py                     # NEW — opt-in mixin, defaults OFF; wraps the
                                     # BtcRegimeGate merge pattern rather than rewriting it
NNNC/
└── NNNC_MLX_Phase*.py               # NEW — one thin subclass per mechanism arm
```

**Structure Decision**: Phases A–C live entirely in a new standalone `regime/` directory and touch exactly one existing file (`utils/custom_indicators.py`, additive). This keeps the go/no-go measurement free of strategy-side risk — no `Framework/` change, no strategy change, nothing that could perturb a running dry run. It also mirrors `btc_corr/`, so the negative result stays reproducible in the same shape as the prior study if the verdict is FAIL. Strategy-side code appears only in Phase D, as an opt-in mixin defaulting to off plus thin per-arm subclasses, so that mixing it in is a no-op until an arm explicitly enables it.

## Execution Phases and the Stop Gate

### Phase A — build the classifier

Instruments at three levels. BTC level reproduces the exp4 taxonomy verbatim as the baseline. Asset level adds the direct trend-versus-reversion measures (variance ratio, Hurst from the VR curve, rolling autocorrelation) alongside existing indicators. Market level adds cross-sectional dispersion, breadth, and median pairwise correlation under the `UNKNOWN` coverage rule.

**Exit condition**: every instrument passes truncated-versus-full-history label identity at three cut points. An instrument that fails is fixed or dropped — it does not proceed on a promise of causality.

### Phase B — characterise and eliminate

Measure the host's own trade duration distribution on W1 **first**, then set the dwell threshold from it. Report distribution, median dwell, and transition matrix per instrument per era. Eliminate instruments below the dwell threshold or materially unstable across eras. Separately, recompute the prior taxonomy's per-state drawdown normalized for time-in-state and state plainly whether the 4× risk discrimination is confirmed or refuted.

**Exit condition**: every instrument carries SURVIVING or ELIMINATED with a measured, quoted reason. If zero instruments survive, the feature stops here and that is the finding.

### ⛔ Phase C — THE STOP GATE

For each surviving instrument, ask whether the phase label at `t` predicts which of the aggressive (`prediction_threshold` 0.55) and conservative (0.80) configurations wins over `t..t+H`, judged on the host's realized trades by risk-adjusted return, versus always using the constant 0.69.

**This is an explicit decision point, not a formality.**

- **FAIL** — no instrument beats the constant in at least two temporally distant windows → **the feature stops.** Write the negative result to `regime/README.md` with the comparison table, commit it, and do not build any mechanism. Per the constitution's experiment gate, this counts as a finding with the same weight as a positive one. The prior art here is a genuine prior: the same test on a different parameter returned NO at every horizon.
- **PASS** — record the winning instrument, level, and horizon `H`. Only then does Phase D begin, and only using that instrument.

### Phase D — mechanisms (unreachable on FAIL)

Strictly ordered, each a separate arm against a common baseline on all three windows:

1. **M3** — phase-conditional `prediction_threshold`. Highest leverage; it is the same lever Phase C validated, so a PASS transfers most directly here.
2. **M1** — phase-conditional trend filter (`entry_trend_filter_enable` / `entry_trend_ema_period`). Cheapest, strongest prior.
3. **M2** — phase-conditional dip guards. Deliberate control arm; expected near-inert, and the R7 liveness measurement is what distinguishes "near-inert" from "measured null".
4. **M4** — phase as model features. **Additionally gated**: only if M3 or M1 showed signal, and Principle IX applies — check phase-feature return correlation against the existing feature set *before* evaluating performance. Requires `Framework CreateScalers` regeneration before any comparison.

Detailed design for Phase D is deliberately not elaborated here beyond what is needed to start it, since Phase C may make it moot.

## Per-Arm Protocol (Phases C and D)

Every arm, without exception:

| Gate | Mechanism |
|---|---|
| Pinned range | `test_strat.sh -t YYYYMMDD-YYYYMMDD`; `-n`/`-o` are forbidden — they compute from today's date |
| Three windows | W1 2025-09-01→2026-08-31 (headline), W2 2024-06-01→2025-05-31, W3 2023-01-01→2023-12-31 (distant) |
| Realistic execution | Volume/liquidity constraints active |
| Liveness proof | Model-condition pass count, full-chain pass count, trade count. Unchanged full-chain count ⇒ **INERT**, results discarded |
| Threshold routing | The parameter is varied via the `<Strategy>.json`, because that file overrides `buy_params` (R5) |
| Sub-1pp deltas | Confirmed under paired repeated runs before acceptance |
| Zero-trade arms | Reported inconclusive, never as a zero-return result |
| Exit invariance | Exit configuration byte-identical between arm and baseline |

## Known Hazards

1. **`NNNC/NNNC_MLX.json` overrides `buy_params` and is untracked.** It is the user's working state. Arms must route threshold changes through it (or a per-arm copy) and must not have it silently rewritten. This is the single most likely way for this feature to produce a false null.
2. **`test_strat.sh` defaults to a today-relative window.** Any arm run without `-t` is not comparable to any other arm.
3. **Cross-sectional breadth is the highest lookahead risk** in the whole study — a dispersion calculation over a panel containing future rows leaks silently and passes casual inspection. R9 validation exists primarily for this.
4. **Dwell threshold must come from the host, not from the momentum dry run.** The 1.4–31h figures circulating in this session belong to a different strategy family (see R6).
5. **A live dry run is currently executing** (`MomentumRegimeBasket15mFast`, since 2026-08-23). Phases A–C touch no strategy code, so they cannot disturb it; Phase D must not modify shared `Framework/` files while it runs.

## Next Command

`/speckit-tasks` — to generate the dependency-ordered task breakdown from this plan.
