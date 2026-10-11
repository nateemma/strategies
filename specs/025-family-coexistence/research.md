# Research: Family Coexistence and Reversion Late-Half Fix (025)

Phase 0. Code facts were read at `321c02d` on 2026-10-11. Each item gives a Decision, a Rationale
and the Alternatives considered.

## R1 — Why the families block each other, and what a rule can change

**Facts (024).**
- freqtrade allows one open trade per pair.
- `max_open_trades` = 10, there are ≤10 pairs, and `slots_full` was 0 in every cell.
- A signal is lost only because its pair is already held.
- About 84% of reversion's lost entries and 68% of momentum's were lost to the OTHER family
  holding the pair: 3,258 of 3,896, and 1,252 of 1,853.

**Decision.** A coexistence rule may only refuse NEW entries, using state known at that bar:
family tags of open trades, and trigger history. Two rule shapes are tested (spec FR-002):
- **A — per-family pair cap.** A family may hold at most `k` pairs at once. A further entry of
  that family is refused while it holds `k`. Capping both families leaves at least `10 − k` pairs
  free of each family at all times.
- **B — momentum-recency guard on reversion.** A reversion entry on a pair is refused if the
  momentum trigger fired on that pair within the last `W` bars (current bar included). The pair
  stays free for momentum's likely follow-on entries, since `breakout_gbb` fires persistently
  while band position is high.

**Alternatives rejected.**
- Multiple open trades per pair (one per family). freqtrade's trade model is one open trade per
  pair; position adjustment merges into one trade and would mix families' exits.
- Evicting an open trade for the other family. Changes the exit model and is out of scope.
- Reversing `FAMILY_PRIORITY`. It decides only 239 same-bar overlaps (024).

## R2 — Where each rule plugs in (strategy-agnostic, no Framework edit)

**Facts.**
- `BaseStrategy.confirm_trade_entry(pair, …, entry_tag, …)` runs at each entry (backtest
  included) and already refuses on volume headroom.
- freqtrade exposes open trades in backtests through `Trade.get_trades_proxy(is_open=True)`.
- Entry rows are built in `SimpleTriggerStrategy.get_entry_conditions` / `_tags`, which already
  holds every trigger's mask per row (024 C4).

**Decision.**
- **Rule A** lives in a new `utils/FamilyCoexistence.py` `FamilyCoexistenceMixin`. It overrides
  `confirm_trade_entry`: first `super()` (the volume check, unchanged), then it refuses when
  `family_of(entry_tag)` already holds `FAMILY_PAIR_CAP` open trades. It is strategy-agnostic
  (reads tags, `FamilyExits.family_of`), and NNNC can adopt it later.
- **Rule B** lives in `SimpleTriggerStrategy._tags`. A per-row mask is computed: "a momentum-family
  trigger fired within the last `W` bars", a causal rolling max over past rows of the momentum
  masks. Rows whose winning family is reversion, and that carry no momentum signal themselves, are
  dropped when the mask is true. Bars where momentum itself fires are unaffected. It is
  vectorised, needs no runtime state, and is reproducible offline.

**Rationale.** No `Framework/` change is needed, so 024's neutrality proof stands. Rule A needs
runtime position state, so it must live in the entry-confirmation hook. Rule B is a pure function
of past bars, so it belongs with the entry mask.

## R3 — Liveness counters (FR-003) reach the run record

**Fact.** `run_arm.py` already sets `FT_ENTRY_CONDITION_SINK` to a scratch directory per run. It
aggregates the per-pair JSON files the strategy writes there (`aggregate_sink` →
`entry_condition_counts`). Values on the strategy object never leave freqtrade's process (024 F2).

**Decision.**
- **Rule B** writes `{pair}_coexist.json` into the sink with
  `{"counts": {"coexist_recency_refused": {"pass": refused_rows, "total": rows}}}`. The existing
  `aggregate_sink` merges it unchanged into `entry_condition_counts`.
- **Rule A** appends one JSON line per refusal to `coexist_events.jsonl` in the sink, with
  `{"family": …, "rule": "pair_cap"}`. A new `aggregate_coexistence(sink)` in `run_arm.py` sums
  them into the record's `coexistence_refusals` = `{rule: {family: n}}`.
- Both write only when the env var is set, so they change no trades.

## R4 — Pre-declared candidate settings (FR-004, constitution VIII)

| rule | neutral | active candidates | note |
|---|---|---|---|
| A pair cap `k` (same for both families) | None (no cap) | **3, 5, 7** | with 10 pairs, 5 is an even split; 3 and 7 bracket it |
| B recency window `W` (bars of 15m) | 0 (off) | **4, 16, 48** (1h, 4h, 12h) | 48 bars ≈ the 12h unclog horizon |

**Selection rule (early folds only).**
1. Compute each active candidate's per-fold wallet sum over f00–f07.
2. Exclude **degenerate** candidates: either family's combined-arm trade count below 25% of its
   single-family count in either half (spec edge case).
3. The highest early sum wins. Ties go to fewer entries lost to the other family.

**Judgement.** The winner is compared with `st_fam_combo` on f08–f14 with the halves paired verdict.
It is adopted only if the late verdict is **IMPROVED**. Otherwise the coexistence stage reports
"no rule adopted", and US2 runs on `st_fam_combo` itself.

## R5 — Reversion tuning steps, inside the combined arm (FR-006, FR-009, Q1 → A)

Current reversion values are ARM_D_OPEN's: `take_profit` 0.022, `unclog_12h_hours` 12,
`atr_stoploss_floor` −0.06 (cap −0.04, multiplier 4.0), and `use_volume_confirmation_stoploss`
True.

| step | parameter | candidates (current excluded) | governs exit reason(s) for liveness |
|---|---|---|---|
| 1 | `take_profit` | 0.015, 0.030, 10.0 (off) | `take_profit` |
| 2 | `unclog_12h_hours` | 6, 24 | `unclog_12h`, `unclog_1d` |
| 3 | `atr_stoploss_floor` (the band's loose edge) | −0.05, −0.08 | `trailing_stop_loss` / `stop_loss` counts, or `trades_hash` change |
| 4 | `use_volume_confirmation_stoploss` | False | as step 3 |

- **Step 3 tunes the floor, not the multiplier.** With cap −0.04 and floor −0.06, `4.0 × atr_pct`
  is clamped for most bars, so a multiplier change would likely be INERT. The floor is the
  binding edge. It stays inside FR-002's "ATR band".
- **Per step:**
  - run each candidate as a combined arm (the chosen coexistence rule, reversion set = previous
    step's set + this candidate);
  - choose by the reversion side's early-fold contribution plus the whole arm's early sum. The
    primary key is the whole-arm early sum, because the deployment metric is the arm, not one
    family;
  - judge the winner against the previous step's arm with the late-half paired verdict, and keep
    it only if IMPROVED.
- **Liveness:** read from the run record's per-family exit counts (024 C8). A change only in the
  reversion family's governed reasons counts.
- The tuned values are applied through `FAMILY_EXIT_OVERRIDES["reversion"]` (024). The momentum set
  is untouched (FR-010).

## R6 — Fingerprint (FR-012)

`FAMILY_PAIR_CAP` and `MOMENTUM_RECENCY_BARS` join 024's optional family members. They are hashed
only when set to a non-neutral value; tuned reversion values are already covered by
`FAMILY_EXITS_RESOLVED`. Arm D, `SimpleTrigger_Gbb` and the 024 arms keep their shas (test).

## R7 — Budget

| stage | arms |
|---|---|
| neutral reproductions (A and B at neutral) | 2 |
| coexistence actives | 6 |
| tuning (3 + 2 + 2 + 1) | 8 |

That is 16 arms × ~5.5 min ≈ 90 min, run as one sequential chain per stage. The stage gates
(choose → judge) need my analysis between stages, so there are 5 chains, not 1.
