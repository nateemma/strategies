# Data Model: Recovery-Aware Reversion Exits (028)

## Pre-registration (`preregistration.json`)

| field | type | rule |
|---|---|---|
| `created` | date | |
| `control` | `{class, plan_tag}` | `SimpleTrigger_RevOnlyVolreg2`, `st_e22_rev2` |
| `candidates` | list of `{id, class, plan_tag, values, applies_recovery_floor, early_gain_pp, late_gain_pp}` | ids C4, C2, C1, C42. Values are frozen. The 2023–26 gains are filled in from the records before commit. |
| `folds` | list of `{fold_id, test_start, test_end}` | e00, e01, e02 (R1) |
| `universe` | list[str] | the 10 `config_wf.json` pairs |
| `fidelity_min` | float | 0.90 |
| `recovery_floor` | float | 0.50 |
| `min_control_trades` | int | 200 |
| `rule` | object | (a)–(d) of R4, written out |
| `deviations` | list | empty at commit. Each later entry is `{date, change, reason}`. |

## Window feasibility (`feasibility.json`)

These are computed per window (2022). The 2023–26 halves come from 027 and are given for context.
- `fidelity`: null replay agreement (overall and by reason) and stop-model agreement.
- `stop_exits`, `recovered_maxhold`, `recovery_share`, `forgone_pp`.
- Per candidate: `replay_gain_pp`, `expected_arm_gain_pp` (= 0.5 × replay), `floor_applies`,
  `pass`.
- `exit_code`: 0 means at least one PASS, 3 means none, 2 means a missing input or failed
  fidelity.

## Candidate verdict (`verdicts.json`)

| field | meaning |
|---|---|
| `id`, `class`, `plan_tag` | identity (FR-006) |
| `sum_pp_arm`, `sum_pp_ctl`, `gain_pp` | rule (a) |
| `paired` | `{verdict, delta_pp, ci_pp_yr, p}` — rule (b) |
| `risk` | `risk.compare` output — rule (c) |
| `early_gain_pp`, `late_gain_pp` | rule (d), copied from the pre-registration |
| `folds_changed` | liveness (count of e-folds whose trades_hash differs from the control) |
| `exit_reasons_arm`, `exit_reasons_ctl` | context |
| `ci_half_width_pp` | power statement |
| `verdict` | ADOPT, NOT ADOPTED, INCONCLUSIVE or STOPPED_AT_FEASIBILITY |
| `failed` | list of the rule letters that failed |

## Exit switches (US3, only if adopted)

| `ExitParams` field | default | meaning | bound |
|---|---|---|---|
| `stop_min_depth` | None (off) | post-grace stop no tighter than this | in [atr_stoploss_floor, 0) |
| `take_profit_atr_mult` | None (off) | take-profit = k × signal-bar ATR, clamped to [0.01, 0.05] | k > 0 |

Fingerprint rule: a None-valued switch field is omitted from the hash.

## State

1. pre-registration committed
2. control run on 2022
3. US1 feasibility, with exit 0, 2 or 3
4. if exit 0: arms run for the passing candidates; any candidate that did not pass is
   STOPPED_AT_FEASIBILITY
5. verdicts
6. promotion, for adopted C1/C2 only
7. report
