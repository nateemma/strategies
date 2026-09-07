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
