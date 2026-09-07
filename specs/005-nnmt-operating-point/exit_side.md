# The exit side (US5 + US6) — the one positive result in this feature

Executed 2026-09-06 after the Phase 5 re-scoping, once the `exit_rvol_threshold`
fix made the parameter expressible. Answers T021–T024, which the T009 gate had
deferred. All 30 arms `returncode` 0.

## Why a 1-D sweep would have produced a meaningless null

`Framework/BaseStrategy.populate_exit_trend` returns EARLY when
`enable_exit_signal` is False:

    if not self.enable_exit_signal.value:
        return dataframe
    ...
    if self.entry_enable_guards.value:
        conditions.append(dataframe["rvol"] > self.exit_rvol_threshold.value)

**NNMT ships with `enable_exit_signal = False`, so `exit_rvol_threshold` is DEAD
CODE on this family.** Sweeping it alone returns identical trades at every value.

Measured, not assumed — exit signal OFF with the gate fully open:

| window | trades | return | vs control |
|---|---|---|---|
| W1 | 176 | +10.708% | **byte-identical trade hash** |
| W2 | 347 | +8.331% | **byte-identical trade hash** |
| W3 | 126 | +1.238% | **byte-identical trade hash** |

So the sweep had to be 2-D: the gate value crossed with the signal flag. A
one-dimensional sweep would have "refuted" research R5 with an artefact.

## The sweep (seed 42 and seed 7, deltas vs each seed's own control)

| exit signal | `exit_rvol` | W1 s42 | W1 s7 | **W1 mean** | W2 mean | W3 mean |
|---|---|---|---|---|---|---|
| ON | 2.0 *(current default)* | −0.13 | −0.11 | **−0.12** | +0.59 | +0.03 |
| ON | **1.0** | +0.74 | +0.81 | **+0.78** | **+2.13** | +0.32 |
| ON | 0.5 | +0.61 | +0.70 | **+0.66** | +0.81 | +0.27 |
| ON | 0.2 *(= entry value)* | +0.09 | — | +0.09 | +0.75 | +0.27 |
| ON | −1.0 *(fully open)* | −0.06 | +0.01 | −0.02 | +1.70 | −0.19 |

**`exit_rvol = 1.0` is best on W1 and W2, and second on W3 by 0.02pp.** Both
seeds agree in sign in 12 of 12 (window × config) cells for that value, with a
between-seed spread of **0.08pp on W1** against a **+0.78pp** effect.

### These two seeds are genuine replicates — checked, not assumed

Every one of the 12 exit arms has a DIFFERENT trade hash between seeds. That
matters because the take-profit arm earlier in this feature did not: there both
seeds produced identical W1 trades, so its "two-seed agreement" was one
measurement reported twice and was recorded as such.

Here the difference is real, and the reason is the substantive finding below.

## The finding that reframes the feature

**NNMT's entry head is inert. Its SELL head is not.**

- **Entry side**: the two seed controls produce **identical** W1 trades. A model
  trained on a different label produces identical W1 trades. A *perfect* entry
  signal is worth −4.56pp (Phase 2).
- **Exit side**: the same two seeds produce **different** trades in every window
  once the exit signal is enabled — because the exits are driven by the
  multi-task sell head, which the seed changes.

The Phase 2 headroom measurement replaced `predict_buy`. It correctly showed the
entry model is worthless and it says **nothing whatever** about the exit model.
That is exactly what research R9 warned: *the oracle bounds what a LABEL can pay,
not how the EXIT harvests it.* R9 is vindicated — and had Phase 5 been abandoned
rather than re-scoped, this result would have been missed.

## Why the exit gate matters, and why 0.2 is the wrong answer

The intuition in R5 was that the exit gate should match NNMT's entry rvol,
because 2.0 against an entry value of 0.2 looked like an accident. That intuition
is **wrong**: `exit_rvol = 0.2` is worth only +0.09pp on W1, nearly as poor as
2.0, and fully open is worse still. The optimum is interior, at 1.0.

Reading it mechanically: the exit gate decides how much volume confirmation an
exit signal needs. At 2.0 almost no exit fires (hence the near-null at the
current default). Fully open lets every weak sell signal fire, including noise.
At 1.0 the model exits on genuine volume expansion only.

## Whether the +10.4pp prior reproduces (FR-017)

**No.** Enabling the exit signal at NNMT's *current* gate is **−0.12pp** on W1,
not +10.4pp.

The prior was measured at an OPEN ENTRY gate. This feature has established that
NNMT cannot profitably occupy that configuration — guards-off costs −34.15pp on
W1 (Phase 3). So the prior is not refuted so much as **inapplicable**: it
described a configuration that is not available to this family. The reproducible
part of it is the direction, and it needed the exit gate moved to appear at all.

## Verdict

**CANDIDATE, not yet promoted**: `enable_exit_signal = True` with
`exit_rvol_threshold = 1.0`, as an NNMT-level override.

Evidence for:

- +0.78pp on W1, the sole acceptance window, positive on all three windows.
- Two genuine seed replicates agreeing to 0.08pp — this satisfies the
  constitution's fifth experiment gate, which requires any accepted delta under
  one percentage point to be confirmed under paired repeated runs.
- An interior optimum with a coherent mechanism, not an edge effect.
- The gain comes from the COMBINATION. The signal alone at the current gate is
  −0.12pp; the gate alone is inert. Neither half works without the other, which
  is why both were previously recorded as nulls.

Evidence against, stated plainly:

- **Not resolved by the single-arm paired bootstrap**: +0.78pp sits inside a
  1.43pp band on W1. The cross-seed and cross-window consistency is evidence the
  single-arm test does not capture, but it is not the same as resolution.
- **The value 1.0 is the argmax of five tested values.** The consistency across
  windows and seeds argues against pure selection effect, but it is a selection
  from a small grid.
- W2's +2.13pp is in-sample and carries no acceptance weight; W3's +0.32pp sits
  on 126 near-flat trades.

**Before promotion** FR-029 requires the capacity curve, since this changes the
exit path and therefore the holding profile. That is T026, which was recorded as
not-applicable when no change was recommended and now becomes applicable again.

---

## Production verification (2026-09-07) — and a W3 sign-flip

After promoting the override, `NNMT_MLX` (production: take-profit **0.022**)
was run with and without it. Trade hashes differ in all three windows, so the
change is live end-to-end.

| window | before | after | Δ |
|---|---|---|---|
| **W1** (acceptance) | 180tr +10.450% | 180tr +11.731% | **+1.281pp** |
| W2 (in-sample) | 353tr +8.833% | 353tr +9.429% | +0.596pp |
| W3 (in-sample, near-flat) | 128tr +2.071% | 129tr +1.323% | **−0.748pp** |

**The "positive on all three windows" claim holds for the CONTROL config
(take-profit 0.032), where it was measured on two seeds. It does NOT hold on
the PRODUCTION config**, where W3 flips negative. That must not be glossed:

| config | take-profit | W1 | W2 | W3 |
|---|---|---|---|---|
| control (2 seeds) | 0.032 | +0.78 | +2.13 | **+0.32** |
| production (1 seed) | 0.022 | +1.28 | +0.60 | **−0.75** |

### The likely mechanism, and it is consistent with the sell-label study

The two configs differ in take-profit, and `regime/SELL_LABEL_PATH.md` measured
exactly the relevant quantity: the share of sell labels PRE-EMPTED by the
take-profit rises as the take-profit tightens — **11.4% at 2.2% against 5.7% at
3.2%**, twice as many. At the tighter production take-profit, more trades would
have closed profitably on their own, so the exit signal's extra exits are more
likely to pre-empt a move rather than save one from a stop.

That is a coherent **interaction between `enable_exit_signal` and
`cexit_take_profit`**, and it was predicted by the label study before this
backtest was run.

### Why the promotion still stands, and what is now open

- **Acceptance is W1 only**, and W1 is positive on both configs — +0.78pp on the
  control across two seeds, +1.28pp on production. W2 is positive on both.
- **W3 is in-sample, near-flat (128 trades, ~+2%), and single-seed here.** A
  −0.75pp move on that base is weak evidence, and the feature has flagged W3 as
  unreliable on this family throughout (research R7).

**OPEN, and recorded rather than acted on:** the exit signal × take-profit
interaction has never been swept jointly. Constitution VIII forbids tuning two
parameter spaces at once on a thin sample, so this is NOT the moment to joint-fit
them. It is the strongest remaining exit-side question, and the sell-label
pre-emption table gives a concrete prior for how it should behave.
