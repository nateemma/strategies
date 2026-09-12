# Data Model: labelfree-meta-labelling

Phase 1. Entities, fields, relationships and validation rules. Types are described by meaning,
not by storage format.

---

## TradeRow

One archived primary trade joined to its entry-time feature vector. **The unit of training.**

| field | meaning | validation |
|---|---|---|
| `fold_id` | which walk-forward fold the trade was taken in | `f00`–`f14` |
| `pair` | instrument | present in the fold's universe |
| `open_date` | entry bar timestamp | inside the fold's test range |
| `close_date` | exit bar timestamp | ≥ `open_date` |
| `profit_abs` | realised profit under real exits, fees and fills | finite |
| `profit_ratio` | realised return fraction | finite |
| `exit_reason` | which exit fired | from the archive |
| `scale_pp_per_abs` | conversion to percentage points | finite, positive |
| `features` | the entry-time feature vector | every component finite; see below |
| `target_win` | `profit_abs > 0` | boolean |
| `target_profit` | `profit_abs` | retained for a future brief; **unused here** — sizing is an explicit Non-Goal |

**Rules**

- Exactly one row per archived trade (FR-001). Rows that cannot be built are **excluded and
  counted**, never imputed (FR-006).
- Two trades on the same pair and bar are **one decision**, not two measurements.
- `target_win` is derived from realised money, never from a forward-excursion stamp.

---

## FeatureVector

Everything knowable at `open_date`. **Pair identity is deliberately absent** (R7).

| feature | source | why it is here |
|---|---|---|
| `primary_confidence` | re-inference (R1) | the model's own probability at the entry bar — the quantity "when to trust the primary" is about |
| `guard_metric` | re-inference | the guard chain's value at entry |
| `bb_width` | re-inference | volatility-band width at entry |
| `rvol` | re-inference | relative volume at entry; the dominant guard |
| `adx` | re-inference | trend strength at entry |
| `atr_pct` | entry bar | normalised volatility |
| `spread_proxy` | entry bar OHLCV | execution cost proxy |
| `quote_volume` | entry bar | liquidity available to the stake |
| `btc_context` | entry bar | market-wide state |
| `time_in_fold` | fold plan | position within the test range |

**Excluded, by design**: `pair`, `pair_group`, and any statistic derived from that pair's own
trade history. The model MUST be unable to rank pairs (FR-005c), so any gain is within-pair and
SC-010 is a genuine test.

**Rules**

- No component may depend on data timestamped after `open_date` (FR-002).
- Causality is **proved per feature** by recomputing with later bars deleted and asserting the
  value is unchanged (FR-003) — not by inspection.
- Components are recomputed by the repo's existing feature code, not reimplemented (FR-005).

---

## ReinferenceRecord

Evidence that the recovered features describe the model that actually traded.

| field | meaning | validation |
|---|---|---|
| `fold_id` | fold re-run | `f00`–`f14` |
| `model_path` | the saved primary model used | exists; unmodified |
| `archived_entries` | `(pair, open_date)` set from the archive | non-empty |
| `reinferred_entries` | `(pair, open_date)` set from re-inference | — |
| `matches_exactly` | set equality | **MUST be true** (FR-005b) |
| `missing` / `extra` | the symmetric difference | reported on failure |

**Rules**

- `matches_exactly == false` **halts the feature** (SC-017). No downstream figure is admissible.
- A mismatch is a finding — models, pinned params or feather data changed since 2026-09-09.

---

## NestedAssignment

Which trades may train the model evaluated on a given fold.

| field | meaning | validation |
|---|---|---|
| `eval_fold` | fold being evaluated | `f05`–`f14` |
| `train_trade_ids` | trades usable for training | every `close_date` < eval fold's `test_start` |
| `n_train` | count | reported beside every per-fold figure |

**Rules**

- Keyed on **`close_date`**, not `open_date` (R6): a trade opening before the boundary but
  closing after it carries an outcome not yet knowable.
- Folds 0–4 are warm-up and are never evaluated (FR-008).
- `n_train` is 91 at `f05`. Thin training sets are reported, not smoothed away.

---

## PersistenceProbe

The W3 distant-era check. **Anti-causal by construction.**

| field | meaning | validation |
|---|---|---|
| `window` | `W3` (2023-01-01 → 2023-12-31) | fixed |
| `trained_on` | later folds | explicitly *after* the tested period |
| `is_anticausal` | always true | must be stated wherever the figure appears |
| `auc` / `gain` | probe results | **never quoted as an out-of-sample return** |

**Rules**

- May never count toward SC-001 (SC-002, SC-018).
- Exists to answer whether the entry-feature/outcome relationship holds ~2.5 years earlier.

---

## VetoDecision

| field | meaning | validation |
|---|---|---|
| `trade_id` | the trade decided on | exists in `TradeRow` |
| `p_win` | secondary model's probability | in [0, 1] |
| `threshold` | operating point | pre-registered |
| `accepted` | `p_win ≥ threshold` | boolean |

---

## VetoResult

One configuration's outcome. **Never reported without its activity metrics.**

| field | meaning | validation |
|---|---|---|
| `threshold`, `seed` | configuration | in the trial log before the result exists |
| `auc` | nested OOS discrimination | vs the 0.55 gate (SC-009) |
| `precision_at_threshold` | were the vetoed trades actually losers | reported beside AUC |
| `retained_trades` | count after the veto | **mandatory** (FR-014, SC-012) |
| `trades_per_year`, `mean_hold_hours` | activity | **mandatory** |
| `return_pp_yr`, `ci` | P&L consequence | vs the **recomputed 4.681 pp/yr** floor |
| `folds_positive` | of 10 | SC-002 needs ≥ 7 |
| `gain_all_pairs` / `gain_ex_zec` / `gain_within_pair` | attribution | SC-010 |
| `inconclusive` | too few retained trades to resolve | if true, the **return is withheld** (FR-015) |

**Rules**

- A configuration retaining too few trades to resolve is INCONCLUSIVE; its return is withheld,
  never reported (FR-015, SC-015). A veto can otherwise "win" by discarding almost everything.
- A fold from which the veto removed every trade has a zero return **by construction**; that is
  not a loss avoided and must not be presented as one.
- The null configuration (threshold accepts everything) appears in the results table as the
  control — never omitted.

---

## TrialRecord

Reuses `research/labelfree/trials.py` unchanged. Written **before** the result exists (FR-011),
carrying the configuration and its pinned capacity parameters. Every configuration counts toward
the deflation statistics, including abandoned and void runs.

---

## Relationships

```
pooled archive ──> TradeRow ──┬──> NestedAssignment ──> VetoDecision ──> VetoResult
                              │                                              │
saved fold models ──> ReinferenceRecord ──(gates)──> FeatureVector           │
                                                                              │
                                              PersistenceProbe (W3, separate) ┘
```

`ReinferenceRecord` **gates** everything downstream: until it matches exactly, no `FeatureVector`
is trustworthy and no `VetoResult` is admissible.
