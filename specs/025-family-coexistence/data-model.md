# Data Model: Family Coexistence and Reversion Late-Half Fix (025)

## CoexistenceRule (class attributes on a SimpleTrigger combined arm)

| attribute | neutral | allowed | rule |
|---|---|---|---|
| `FAMILY_PAIR_CAP` | `None` | int 1..9 | Rule A. Max open pairs per family, the same for every family. Enforced at entry confirmation; never closes a trade. |
| `MOMENTUM_RECENCY_BARS` | `0` | int ≥ 1 | Rule B. Drop a reversion-only signal row if a momentum-family trigger fired on that pair within the last N bars (current included). While active, an overlap row (both families fire) is tagged momentum (`+ov`), not reversion (G2). |
| `FAMILY_EXIT_OVERRIDES` (024) | `{}` | per family, `ExitParams` fields | MERGED along the MRO (ancestors first, per family, per key; G1), so a subclass adds to its parent's overrides instead of replacing them. |

- At neutral, both are no-ops, so the arm must reproduce `st_fam_combo` exactly (SC-001).
- Validation at class creation: `FAMILY_PAIR_CAP` must be an int in 1..9 or None.
  `MOMENTUM_RECENCY_BARS` must be an int ≥ 0.
- Rule B needs at least one momentum trigger in `TRIGGERS`. Otherwise it is refused, because it
  would be inert by construction.

## CoexistenceCounts (run record, from the sink; R3)

| field | source | meaning |
|---|---|---|
| `entry_condition_counts.coexist_recency_refused` | rule B per-pair sink file | `pass` = signal rows dropped by rule B; `total` = rows |
| `coexistence_refusals` | rule A `coexist_events.jsonl` | `{"pair_cap": {family: n_refused}}` |

## Candidate set (pre-declared, R4/R5)

| stage | parameter | neutral/current | candidates |
|---|---|---|---|
| coexistence | `FAMILY_PAIR_CAP` | None | 3, 5, 7 |
| coexistence | `MOMENTUM_RECENCY_BARS` | 0 | 4, 16, 48 |
| tune 1 | reversion `take_profit` | 0.022 | 0.015, 0.030, 10.0 |
| tune 2 | reversion `unclog_12h_hours` | 12 | 6, 24 |
| tune 3 | reversion `atr_stoploss_floor` | −0.06 | −0.05, −0.08 |
| tune 4 | reversion `use_volume_confirmation_stoploss` | True | False |

## StageDecision (recorded in `specs/025-family-coexistence/decisions.md`)

| field | meaning |
|---|---|
| `stage` | coexistence / tune1..tune4 |
| `early_table` | per candidate: whole-arm f00–f07 sum, reversion-side early sum, lost-to-other early, degenerate flag |
| `chosen` | the early winner, or none |
| `late_verdict` | the halves late verdict of chosen vs previous (IMPROVED / UNCHANGED / …) with CI and p |
| `adopted` | `true` only if `late_verdict == IMPROVED` |
| `liveness` | the governed exit-reason counts (tuning) or refusal counts (coexistence), against the previous arm |
