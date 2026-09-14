# Implementation Plan: Single-Pair (LINK) Variant of NNNC_MLX

**Branch**: `008-single-pair-variant` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)
**Input**: 21 FRs, 8 SCs, 4 user stories, 6 edge cases, checklist 16/16

## Summary

Train `NNNC_MLX_LINK` — identical to `NNNC_MLX` in gate, exits, label and architecture, differing
only in seeing one pair — and compare it against the 11-pair model **on LINK trades only**.

The deliverable is an answer to "does cross-pair training help, harm, or not matter for LINK
selection", **not** a P&L verdict. At 8 W1 LINK trades against a 7.633pp noise floor, returns
cannot resolve; trade selection and per-trade expectancy can.

## Technical Context

**Runtime**: Python 3.12 via the repo `.venv`; zsh from the repo root.
**Reused unchanged**: `regime/run_arm.py`, `regime/tail/compare.py`, `evaluation/`,
`scripts/test_strat.sh`, `Framework/*`.
**New**: one strategy class, one params file, one config, one analysis script.
**Constraints**: sequential only; no Framework/harness modification; variant differs from control
in pair universe alone.

### Decided: how the variant is isolated

| concern | decision | why |
|---|---|---|
| separate model | new class `NNNC_MLX_LINK(NNNC_MLX)`, body `pass` | `get_model_path` keys off `self.__class__.__name__`, so a distinct class gets its own `saved_data/` automatically |
| identical gate | **full copy** of `NNNC_MLX.json` with `strategy_name` rewritten | the tuned gate lives in a FILE and is NOT inherited; five strategies were found ungated on 2026-09-09 this exact way |
| identical exits | the same copy carries buy AND sell blocks | a buy-only file silently reverts exits to BaseStrategy defaults — measured at −2.4pp the day before |
| one pair | `config/config_link.json`, whitelist `["LINK/USDT"]` | `run_arm.py --config`, `test_strat.sh -c` |

**Verification, not assumption**: the ungated-gate warning must be absent from every run
(FR-004), and the model artefact must exist (FR-005). Both are checked, not inferred.

### Decided: the control is not retrained

The existing `nnnc_v2` arms are the control (W1 99tr +9.831%, W2 196tr, W3 62tr). Retraining the
11-pair model would change two things at once and cost 15 minutes for nothing.

### Decided: analysis is pair-restricted, and that is the whole point

Both arms' trades are filtered to LINK before anything is computed. A whole-portfolio comparison
would be meaningless — the variant has no other pairs to trade.

### Compute budget

| phase | work | cost |
|---|---|---|
| A | build class, params, config; validate config | ~0 |
| B | train seed 1 | ~12 min |
| C | W1/W2/W3 arms | ~3 min |
| D | pair-restricted analysis (US2, US3) | ~0 |
| E | seed 7 + rerun arms (FR-016) | ~15 min |
| F | pooled folds (US4, P3) — **only if B–D justify it** | ~1.5 h |

Phases A–E are **~30 minutes**. Phase F is gated on the earlier result: if US2 shows the two models
select identical LINK trades, pooled folds cannot add anything and are skipped.

## Constitution Check

| principle | status | note |
|---|---|---|
| I. Evaluate in the Deployment Regime | **PASS** | W1 is the verdict window; W2/W3 reported and labelled. |
| II. Persistence Across Distant Eras | **PASS WITH CAVEAT** | LINK contributes in all three windows (11.6/9.5/26.4%), which is *why* it was chosen over ZEC. But 8 and 5 trades in W1/W3 cannot demonstrate persistence of a *delta*; only the per-trade expectancy comparison has any power, and that is reported per window. |
| III. Learnability Is Not Edge | **PASS** | FR-015: no classifier metric feeds a verdict. |
| IV. Verify Parameters Are Actually Live | **PASS** | FR-004's warning check is exactly this, and it exists because the failure happened. |
| V. Lookahead-Clean Is Not Realistic | **PASS** | Execution model untouched; volume/liquidity constraints unchanged. |
| VI. Diagnose Before Tuning | **PASS** | Nothing is tuned; the rotation table was measured before the feature was scoped. |
| VII. Validate Cheaply | **PASS** | Phases A–E cost ~30 min and gate the 1.5 h phase F. |
| VIII. One Parameter Space at a Time | **PASS** | One thing changes: the pair universe. |
| IX. Orthogonality | **N/A** | No factor added. |
| X. Structural Fixes, Not Per-Pair Overrides | **PASS WITH INTERPRETATION** | See below. |

### Principle X — this feature is a per-pair model, which the principle warns against

Principle X prohibits **per-pair overrides inside a strategy** — special-casing a pair so the
shipped system behaves differently for it. This feature builds a *separate model* to answer a
measurement question, and its stated Non-Goals forbid building variants for further pairs.

**The principle constrains the outcome, not the experiment.** If LINK-only wins, that does NOT
license shipping a per-pair model; it would mean cross-pair training dilutes, and the structural
response would be a change to how one model trains, not eleven models. The conclusion (T-final)
must state this explicitly so a positive result cannot be mistaken for a licence to special-case.

## Project Structure

```
specs/008-single-pair-variant/
├── spec.md  plan.md  research.md  data-model.md
├── contracts/analysis-cli.md
├── quickstart.md
└── checklists/requirements.md

NNNC/NNNC_MLX_LINK.py          # generated-simple: subclass, body `pass`
NNNC/NNNC_MLX_LINK.json        # FULL copy of NNNC_MLX.json, strategy_name rewritten
config/config_link.json        # whitelist ["LINK/USDT"]
regime/pair_restricted.py      # NEW — the only new analysis code (US2, US3)
```

**Structure decision**: one new analysis module. Everything else is an existing tool invoked with
different arguments. `regime/pair_restricted.py` exists because no current tool filters two arms'
trades to one pair and reports the set comparison — `compare.py` compares whole arms, and
`trade_forensics.py` aggregates by exit reason.

## Complexity Tracking

| item | why needed | simpler alternative rejected because |
|---|---|---|
| A new strategy class | `get_model_path` keys off the class name; without it the variant would overwrite the 11-pair model's weights | Reusing `NNNC_MLX` with a different config would train into the same directory and destroy the control |
| A full params-file copy | file params are not inherited, and a buy-only file reverts the exit stack | Relying on class defaults is exactly the ungated-strategy defect found the previous day |
| `regime/pair_restricted.py` | no existing tool does a pair-restricted set comparison | Extending `compare.py` would fork the statistic of record; it is fed, not modified |
