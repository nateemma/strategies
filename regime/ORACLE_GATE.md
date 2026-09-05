# The Oracle Gate — standing process for label and exit changes

**Status**: adopted 2026-09-05 (feature 004). Applies to any strategy family in
this repository, not only NNNC.

## Why this exists

A perfect-prediction oracle measured the CEILING of NNNC's execution stack and
found the whole span from *no model at all* to a *perfect model* to be 1.45pp on
W1 — of which the real model already captures 0.78pp, leaving 0.676pp for any
possible improvement. Six model-side changes before that measurement had all
been absorbed or inert, at 8–45 minutes of training each. **There was never more
than a point on the table, and it cost many retrains to discover.**

The oracle needs no training. It answers, in minutes, whether a proposed change
could help at all.

## The gate

Before training a model on any new label or exit configuration, run the
configuration through the oracle with the guard chain neutralised and apply:

| verdict | condition |
|---|---|
| **PASS** | W1 guards-off oracle return **> 0** AND **`n_trades` >= 200** |
| **INCONCLUSIVE** | trades identical to the control (the change was absorbed), or `returncode != 0` |
| **FAIL** | anything else — *including an improvement that remains negative* |

**A FAIL records progress but does NOT license training.** The premise is that
until perfect foreknowledge can make money without the guards, there is no edge
for a model to learn, so a better model cannot pay.

### Why "positive", not "improved"

An improvement from −18.89% to −12% is real information and worth recording. It
is not a reason to spend a retrain, because the configuration still describes
trades that lose money when predicted perfectly.

### Why the 200-trade floor

**This is a JUDGEMENT, not a measurement, and is recorded as such so it can be
argued with.** Guards-on arms hold ~95 trades. A guards-off variant that reaches
profitability by collapsing toward that count has not fixed the label — it has
re-imposed a gate, and the gate was never the thing under test. 200 is roughly
twice the guards-on count: low enough not to force a particular solution, high
enough that a variant cannot pass by trading almost nothing.

## Two rules that keep the gate honest

**1. Realised P&L only. Never a classifier metric.** `val_mcc`, macro-precision,
EV-per-signal and learnability MCC have each failed to predict realised return in
this repository — EV pointed the wrong way by 43pp, precision tripled while
returns halved, and MCC fell while returns rose. Record them as context if
useful; they may never be an acceptance input.

**2. Every oracle figure is a LOOKAHEAD CEILING.** The oracle's entry signal is
computed from future bars. Its numbers are a diagnostic bound, never an
achievable return, and **no oracle configuration may be promoted toward live
trading under any result.** Report them labelled, always.

## How to run it

Guards-off is expressed by NEUTRALISING the six quality thresholds, never by
`--guards off` — that would also remove the `volume > 0` data-sanity check,
which sits in the same all-or-nothing block:

    NEUT="--set entry_rvol_threshold=-1.0 --set entry_atr_pct=-1.0 \
          --set entry_adx_threshold=-1.0 --set entry_bb_width_threshold=-1.0 \
          --set entry_guard_threshold=5.0 --set entry_close_norm_threshold=5.0"

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/run_arm.py --strategy NNNC_Oracle \
      --window W1 --tag <variant> $NEUT [--set-sell ...] [--roi ...]

Out-of-range values are honoured, not clamped (verified: `rvol=20` → 0 trades),
and `guard_metric`/`close_norm` are bounded in [−1, 1], so 5.0 neutralises them
completely.

**`regime/tail/compare.py` does not produce this verdict.** It emits its own
four-way classification on a 1pp power floor calibrated for ~97-trade arms; on a
1760-trade arm it would call a +2pp move "improved" while the arm is still deeply
negative. Use it for liveness (trade hashes), trade counts and in-sample
labelling — then apply the gate above by hand.

W1 is the sole acceptance window. W2 and W3 fall inside the training window; for
oracle arms no weights are fitted, so they may CORROBORATE a mechanism, but they
never confer acceptance.

## Worked example (T016)

Applied to the cheapest variant addressing feature 004's diagnosis — the exit
payoff ratio. The diagnosis found the stack caps winners at `cexit_take_profit`
= +2.4% while letting losers run to −4%..−6%, a ~0.45:1 ratio that only survives
at the ~75% win rate the guard chain manufactures. The prediction under test is
that raising the cap moves the oracle's guards-off return directly.

The result of that application is recorded in
`specs/004-oracle-label-exit-alignment/screening.md`; the reference point is the
unmodified baseline, `oracle_nog` W1 = **1760 trades, −18.89%**.
