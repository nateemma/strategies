# Contract: Deployable wrapper (`NNDP/NNDP_MLX.py`)

A thin `BaseNNStrategy` subclass that lets the **arbiter** score the trained policy as a real
strategy. It is the only production-shaped artefact this feature creates.

## Required behaviour

| Requirement | Contract |
|---|---|
| FR-027 | `get_predictions` loads the trained policy and returns exposure `w_t`; entry when `w_t` crosses the band's upper edge, exit when it crosses the lower edge |
| FR-028 | The ATR stop and stoploss-grace logic are left **unchanged** from the base, so the comparison with the control isolates the entry/exit signal |
| FR-029 | `get_model_path` is pinned explicitly (AGENT_GUIDE silent-failure note) |
| Constitution | Parameter dictionaries MUST be **merged** from the parent, never redefined as fresh literals — a fresh dict silently drops inherited keys and previously disabled stoploss grace undetected |
| FR-002 | Features are the pair-agnostic include list; no pair-specific branches anywhere |

## Consequence that must be measured, not assumed

The policy is trained on band-crossing exits only (FR-014a), but this wrapper runs with the
deployment exit stack active. FR-014b and SC-018 therefore require reporting, for the candidate:
simulator P&L, arbiter P&L, their difference, and the share of trades whose exit was decided by the
deployment stack rather than the band. Acceptance is blocked until that reconciliation exists.

## Evaluation path

The wrapper is driven through the existing harness — `regime/wf/run_folds.py --plan <plan> --seed
<n>` — then pooled with `regime/wf/pool.py` and compared against the control with
`regime/wf/compare_pooled.py --arm <arm> --control regime/wf/pooled_nnnc.json`.
