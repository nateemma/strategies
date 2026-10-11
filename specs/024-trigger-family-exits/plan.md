# Implementation Plan: Trigger-Family Exit Presets

**Branch**: `024-trigger-family-exits` (spec directory; commits go to master) | **Date**: 2026-10-11 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/024-trigger-family-exits/spec.md`

## Summary

**What it does.** Each registered trigger gets a family (reversion / momentum / unassigned).
Each family gets an exit-parameter set layered on arm D's framework. Trades carry their family
in `enter_tag` from entry to exit. A combined arm runs `gbb` (reversion) and `breakout_gbb`
(momentum) together, with reversion priority on overlap bars, and reports what each family lost
to contention.

**How it works.**
- One additive framework hook: `BaseStrategy._exit_params(trade=None)`. Every exit and stop
  decision already reads its numbers there, and `ExitParams` already carries the unclog hours
  (research R1).
- A strategy-agnostic `FamilyExitMixin` overrides that hook by tag (C2).
- A new preset `ARM_D_FAMILY` turns families on. Existing presets stay byte-identical (C4).

**Correctness gates.** The family arms must reproduce 023's `st_gbb` and `st_bgo_mx`
trade-for-trade (SC-001). Each side of the combined arm must reproduce its single-family arm
(SC-002). Per-family tuning is deferred (Q1 → A).

## Technical Context

**Language/Version**: Python 3.12 (`.venv` at the repo root; conda forbidden)

**Primary Dependencies**: freqtrade (backtesting, `enter_tag` → `trade.enter_tag`), pandas,
numpy. The 023 harness: `evaluation/{foldplan,run_folds,pool,compare_pooled,halves,fingerprint}`,
`regime/run_arm.py`, `research/outcomes/{triggers,causality}`

**Storage**: JSON arm records (`regime/arms/`), plans (`evaluation/folds/`), and backtest zips
(`user_data/backtest_results/`)

**Testing**: pytest (`Framework/tests`, `evaluation/tests`, `SimpleTrigger/tests`,
`research/outcomes/tests`). The walk-forward reproduction by `trades_hash` is the acceptance test.

**Target Platform**: macOS / Apple Silicon. No model training in this feature.

**Project Type**: research codebase (strategy family + evaluation CLI tools)

**Performance Goals**: one 15-fold arm ≈ 4–6 min. The total backtest budget is about 8 arms
(~45 min), run sequentially.

**Constraints**:
- `Framework/` stays strategy-agnostic.
- Parameter dicts merge, never replace.
- One backtest chain at a time.
- Existing presets and every 023 recorded arm stay reproducible.
- Every result is PROVISIONAL (gate 2).

**Scale/Scope**: 3 new modules or files and 5 edited ones (see Project Structure). About 6 new
strategy configs and 8 walk-forward arms.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design. The result is
unchanged.*

| principle | status | how |
|---|---|---|
| I. Deployment regime | PASS | Same 2023-01 → 2026-09 walk-forward span as 023. Halves are reported. |
| II. Distant-era persistence (NON-NEGOTIABLE) | PASS, with every result labelled PROVISIONAL | No distant era exists. Every verdict carries the gate-2 label, and nothing is promoted. |
| III. Learnability ≠ edge | N/A | No model. Judged on walk-forward P&L only. |
| IV. Parameters proven live | PASS | SC-001 reproduction plus exit-reason counts (momentum `unclog_12h` matches `st_bgo_mx`'s). `untagged_trades` (from the archive) = 0. Stop-path reads are routed through params so stop overrides cannot be silently inert (R2). |
| V. Realistic execution | PASS | The harness is unchanged (volume headroom, stake capping). |
| VI. Diagnose before tuning | PASS | Tuning is deferred. The contention report is the diagnostic that will scope it. |
| VII. Validate cheaply first | PASS | 023 Addendum 3 was the cheap check (`st_bgo_mx`). |
| VIII. One parameter space at a time | PASS | No tuning in 024. FR-012 binds the follow-up. |
| IX. Orthogonality | PASS | The per-fold correlation between families is reported (023: 0.39–0.54). |
| X. No per-pair code | PASS | Families are keyed by trigger, never by pair. |
| XI. No leakage | PASS | No fitted transforms. Every trigger passes the 023 causality gate per name. |
| Framework strategy-agnostic | PASS | `Framework/` receives only a `trade` pass-through and params-routed stop reads (C1). The family logic lives in `utils/FamilyExits.py` and `SimpleTrigger/`. |
| Params merge from parent | PASS | Family sets resolve via `dataclasses.replace` over the preset. Subclass `FAMILY_EXITS` merges per family. |
| Repo boundary | PASS | All edits are under `user_data/strategies`. |

## Project Structure

### Documentation (this feature)

```text
specs/024-trigger-family-exits/
├── plan.md              # this file
├── research.md          # R1–R10
├── data-model.md        # families, exit sets, tags, combined-arm config, contention record
├── quickstart.md        # validation runs 0–5
├── contracts/
│   └── interfaces.md    # C1–C8
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (`user_data/strategies`)

```text
Framework/BaseStrategy.py            # EDIT C1: _exit_params(trade=None); 3 call sites pass trade;
                                     #   stop-path reads routed through params (R2)
utils/FamilyExits.py                 # NEW C2: FamilyExitMixin, tag parse, untagged counter
utils/StopInvariants.py              # EDIT C3: bound read from _exit_params(trade)
research/outcomes/triggers.py        # EDIT C5: Trigger.family / family_reason; --list shows it
research/outcomes/{builtin,audit}_triggers.py  # EDIT C5: assign families at registration
SimpleTrigger/SimpleTriggerStrategy.py         # EDIT C4: TRIGGERS / DISABLED_TRIGGERS /
                                               #   FAMILY_PRIORITY / FAMILY_EXITS; tag rewrite
SimpleTrigger/presets.py             # EDIT: ARM_D_FAMILY preset + family exit sets
SimpleTrigger/SimpleTrigger_Fam*.py  # NEW: FamGbb, FamBgo, FamCombined (+ 2 disabled-side variants)
evaluation/fingerprint.py            # EDIT C7: optional family members
evaluation/contention.py             # NEW C6
regime/run_arm.py                    # EDIT C8: untagged_trades, family/overlap counts, per-family exit counts (from the archive)
*/tests/                             # NEW tests for C1–C8 (listed in tasks)
```

**Structure Decision**: same layout as 023. Families are a SimpleTrigger capability built on a
strategy-agnostic mixin in `utils/`, so a later NNNC feature can adopt it (FR-015) without
touching `Framework/` again.

## Implementation Phases (for /speckit-tasks)

1. **Foundation (no behaviour change):**
   - the C1 hook and the R2 routing;
   - neutrality proven on recorded arms (quickstart 0);
   - the C5 registry families and their test (SC-006).
2. **US1:**
   - the C2 mixin, the `ARM_D_FAMILY` preset and the C4 single-trigger path;
   - C7 fingerprint coverage;
   - `st_fam_gbb` and `st_fam_bgo` reproduce `st_gbb` and `st_bgo_mx` (SC-001).
3. **US2:**
   - combined entry and overlap tagging (C4);
   - disabled-side reproduction (SC-002);
   - the full combined arm;
   - C6 contention;
   - comparison and report (SC-005).
4. **Polish:** the report, memory, and the full test gate.

## Risks

| risk | mitigation |
|---|---|
| The `_exit_params` edit changes a recorded arm | Hash-checked reproduction of `st_gbb` f00, `wf_aro` f00, `st_bbo` f00 and `stl_bo` f00 before any family work |
| The momentum params path diverges from `HoldWindowMixin` | Case analysis (R5) plus SC-001 exact reproduction of `st_bgo_mx`. Any divergence blocks US2. |
| The tag is lost or rewritten by freqtrade | `untagged_trades` computed from the archive (C8); SC-007 requires 0 |
| Combined-arm overlap order is non-deterministic | Fixed `FAMILY_PRIORITY` and fixed trigger order, and a unit test on a synthetic frame |
| The bootstrap's set-iteration nondeterminism | `PYTHONHASHSEED=0` on every comparison (the open user decision from 023 is unchanged) |

## Complexity Tracking

None. There are no constitution violations to justify.
