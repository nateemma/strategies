# T026 / FR-029 — capacity curve for the exit-side candidate

Required before promotion because the candidate changes the exit path. Measured
by raising `QUOTE_VOLUME_HEADROOM_MULT` on a subclass: `custom_stake_amount`
caps a trade's stake at 1/MULT of the candle's quote volume, so MULT 20 ≈ a
2× book and MULT 40 ≈ 4× against the same market — the capacity curve without
touching the wallet, the config or the model. 12 arms, all `returncode` 0.

Production = NNMT as shipped (exit signal OFF).
Candidate = `enable_exit_signal=True` + `exit_rvol_threshold=1.0`.
Both on `NNMT_Ctl_s42`, so they differ in the exit configuration alone.

## The curve

| window | book | production | retain | candidate | retain | advantage |
|---|---|---|---|---|---|---|
| **W1** (acceptance) | 1× | +10.71% | 100% | +11.45% | 100% | **+0.74pp** |
| | 2× | +10.40% | 97% | +10.93% | 96% | +0.53pp |
| | 4× | +9.08% | **85%** | +9.30% | **81%** | **+0.22pp** |
| W2 (in-sample) | 1× | +8.33% | 100% | +10.52% | 100% | +2.18pp |
| | 2× | +7.26% | 87% | +8.80% | 84% | +1.54pp |
| | 4× | +6.06% | 73% | +7.17% | 68% | +1.11pp |
| W3 (in-sample, near-flat) | 1× | +1.24% | 100% | +1.52% | 100% | +0.28pp |
| | 2× | +1.06% | 86% | +1.09% | 72% | +0.02pp |
| | 4× | +0.92% | 74% | +0.92% | 61% | −0.00pp |

Trade counts are constant across the ladder (176 on W1 at every book size), so
this is entirely a stake-size effect: the same entries, filled smaller.

## Two readings, and they point opposite ways

**Good news — no capacity cliff, and far better than the NNNC precedent.** At a
4× book the candidate still retains **81%** of its W1 return. The comparable
NNNC open-guards candidate retained only **57%** against its production's 74%.
NNMT's candidate is materially more capacity-robust: a 4pp retention penalty
against NNNC's 17pp.

**Bad news — the advantage decays with book size.** The candidate loses return
FASTER than production in all three windows (W1 81% vs 85%, W2 68% vs 73%,
W3 61% vs 74%), so the edge erodes as the book grows:

    W1 advantage:  +0.74pp (1x)  ->  +0.53pp (2x)  ->  +0.22pp (4x)
    W3 advantage:  +0.28pp (1x)  ->  +0.02pp (2x)  ->  -0.00pp (4x)

The mechanism is straightforward: the candidate's gain comes from better exits
on a subset of trades. When the stake cap bites, those trades are filled smaller,
so the absolute benefit shrinks faster than the baseline return does.

W3's retention figures (72%, 61%) look alarming but rest on a +1.24% base over
126 trades — percentages of a near-flat number. They should not carry weight.

## What this does to the verdict

The exit-side result was already **not resolved** by its single-arm paired
bootstrap: +0.78pp on W1 inside a 1.43pp band, rescued only by cross-seed and
cross-window consistency. The capacity curve makes the case book-dependent:

- At the **current book**, the candidate is the best-supported change in this
  feature: +0.74–0.78pp on W1, positive in 3/3 windows, 2/2 genuine seed
  replicates agreeing to 0.08pp, and no capacity cliff.
- At a **4× book** the W1 advantage is +0.22pp — comfortably inside noise, and
  gone entirely on W3.

**Recommendation: promote at the current book, and record the decay.** The
change is safe (no cliff, 81% retention at 4×) and is the only positive result
on this family. But it is NOT a change that scales, and the promotion note must
say so, so that a future book increase triggers a re-measurement rather than an
assumption that the edge came along.

This is the reverse of the usual concern. The normal capacity question is
whether a candidate is an illiquidity mirage that vanishes under realistic
sizing. This one survives realistic sizing — it just stops being worth much.

---

## Addendum — would MORE, SMALLER positions use the book better?

The natural response to a stake cap is to split the size across more concurrent
positions. Measured on the W1 control, the answer for NNMT is **no**, and the
reason identifies what the real constraint is.

    config           stake_amount 900   max_open_trades 10   balance 10,000

    SLOTS      peak concurrent = 9 of 10     -> the slot limit NEVER binds
               time-weighted mean concurrent = 0.090
               histogram {0:86, 1:113, 2:49, 3:34, 4:24, 5:17, 6:9, 7:9, 8:8, 9:3}

    STAKE CAP  binds on 110/176 = 62.5% of trades
               achieved vs configured 900:  p10 16%   p25 30%   p50 57%   p75 100%

    DEPLOYED   average capital in the market = 0.50% of the wallet
               -> the strategy is ~99.5% in CASH

**Raising `max_open_trades` cannot help: it never binds.** Peak concurrency is
9 of 10 slots, so no signal was ever rejected for want of a slot. There is no
queue of trades waiting to be admitted.

**Lowering `stake_amount` would strictly hurt.** 37.5% of trades already fill at
the full 900; capping them lower would shrink those for no gain. The cap only
binds where candle volume is thin, and a smaller configured stake does not
create volume.

**Splitting one order across two positions does not evade the cap**, because the
cap is per-candle-volume. Two 450 entries in the same candle face the same
constraint as one 900. It only helps across DIFFERENT pairs simultaneously —
which requires simultaneous signals, and mean concurrency is 0.090.

### What the numbers actually say

At the current book the cap costs about **10%** of potential return (realised
+10.71% against a linear-scaling estimate of +11.87% uncapped). That is the
honest size of the liquidity drag today — smaller than the 62.5% binding rate
suggests, because the trades that get capped hardest are not the ones carrying
the return.

The deeper point is the 0.50% deployment figure. **NNMT is not capital-limited,
it is SIGNAL-limited.** It cannot use the book it already has: 176 trades a year
with a 3-hour median hold leaves the wallet idle 99.5% of the time. So the route
to a larger book is more signals — more pairs, or a gate that admits more trades
— or a deeper venue (`regime/EXCHANGE_VENUES.md` measures Kraken at 52x
Binance.US depth on exactly these pairs). It is not smaller stakes, and it is
not more slots.

This also reframes the capacity curve above: the decay from 100% to 85% at a 4x
book is a stake-cap effect on a strategy that is already 99.5% in cash. Fixing
it by sizing is not available; the levers are venue depth and signal count.
