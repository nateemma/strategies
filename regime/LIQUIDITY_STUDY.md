# Liquidity study — NNNC_MLX_Open vs production

Run 2026-09-06 ahead of promoting `NNNC_MLX_Open` to default. Three questions:
are the fills real, how much capital actually gets deployed, and does the
configuration scale.

## 1. The fills ARE real — no phantom returns

`confirm_trade_entry` requires candle quote volume >= `max(1000, 10 x trade
quote size)`, and `custom_stake_amount` CAPS the stake at 1/10 of that volume
rather than rejecting. So every filled trade in both arms is <=10% of its
candle by construction, and the phantom-fill failure mode this repo has hit
twice does not apply here.

**Note this changes what "liquidity risk" looks like**: it appears as SHRUNKEN
STAKES, not as missing or impossible trades.

## 2. The candidate is far more volume-constrained

Configured stake 900 on a 10,000 wallet:

    arm            n   median   mean    p10     p90  at full  <10% of full
    PRODUCTION   332    899.8  679.8  178.0   900.0    56.0%          0.0%
    CANDIDATE   1489    418.5  503.5  128.0   900.0    31.4%          0.0%

    arm          total staked  total P&L  return on DEPLOYED capital  wallet ret%
    PRODUCTION        225,692     2412.8                       1.07%       24.13%
    CANDIDATE         749,699     3409.6                       0.45%       34.10%

**Only 31% of the candidate's trades get full size against production's 56%,
and it earns less than half the return per unit of capital deployed.** Its
higher wallet return comes from recycling the same capital 4.5x more often, not
from better trades.

## 3. It does not scale as well — the capacity curve

Raising `QUOTE_VOLUME_HEADROOM_MULT` caps the stake at a smaller fraction of the
candle, which is equivalent to trading a proportionally larger book against the
same market. No retraining; model aliased so the cap is the only difference.

    arm                        W1 ret%  W2 ret%  W3 ret%   summed   retention
    PRODUCTION  mult 10 (1x)     10.45    10.31     3.37    24.13        100%
    PRODUCTION  mult 20 (2x)      9.10     9.16     2.34    20.60         85%
    PRODUCTION  mult 40 (4x)      7.99     8.14     1.75    17.88         74%

    CANDIDATE   mult 10 (1x)      6.94    16.91    10.25    34.10        100%
    CANDIDATE   mult 20 (2x)      7.22    11.63     6.48    25.33         74%
    CANDIDATE   mult 40 (4x)      7.05     9.17     3.07    19.29         57%

**The candidate retains 57% of its return at a 4x book against production's
74%**, and the gap between them closes from 9.97pp to 1.41pp — which is well
inside the 6.38pp seed standard deviation, i.e. gone.

The detail matters: the candidate's **W1 is flat across scales** (6.94 / 7.22 /
7.05). All of the decay is in W2 and W3 (16.91 -> 9.17, 10.25 -> 3.07). Its
advantage over production lives specifically in the thin-liquidity opportunities
that do not survive a bigger book.

## Verdict for promotion

**At the CURRENT book (10,000 wallet, 900 stake) the configuration is sound.**
Fills are realistic, nothing is phantom, and the measured behaviour is what the
backtests say it is.

**Two qualifications, both material:**

1. **It is a small-book configuration.** Its edge over production is
   substantially a low-capacity edge, and disappears by a 4x book. If the wallet
   grows, this choice should be re-measured, not assumed.
2. **Combined with the seed result** (mean +27.20 vs +24.13, sd 6.38), the
   candidate's return advantage at ANY scale is inside noise. The case for it
   remains headroom and measurability, not performance — see
   `regime/CANDIDATE_STATUS.md`.

Neither is a reason not to promote it as the research/default configuration.
Both are reasons not to expect more money from it.

---

# Appendix — is the 1/10 fill cap justified? No, it is a chosen constant

`QUOTE_VOLUME_HEADROOM_MULT = 10.0` with the comment "10x ~ <=10% market impact
target". Never derived. The only prior sweep (momentum basket, 2026-08) went
TIGHTER and found tightening improved PF (1.64 -> 1.83) and drawdown
(58.5% -> 36.3%). The looser direction had never been measured. It has now:

    setting                        W1      W2      W3   summed  worstDD  meanPF  ret/DD
    mult 3  (33% of candle)      8.70   25.73   16.54    50.97    10.46    1.34    4.87
    mult 5  (20% of candle)      8.20   21.38   13.68    43.25    10.51    1.32    4.12
    mult 10 (10%) DEFAULT        6.94   16.91   10.25    34.10     8.54    1.32    3.99
    mult 20 (5%)                 7.22   11.63    6.48    25.33     7.46    1.32    3.39
    mult 40 (2.5%)               7.05    9.17    3.07    19.29     5.70    1.33    3.39

Loosening to 20% is +27%, to 33% is +50%. Returns scale almost linearly.

**But mean profit factor is FLAT at ~1.32 across the entire range.** Trade
quality does not change at all — only the capital deployed per trade. That is
the signature of a pure sizing artefact, not an edge.

And the backtest has **no slippage model**: `entry_pricing` uses
`order_book_top = 1`, so fills occur at the touch regardless of order size.
Taking 33% of a 15m alt candle at the touch is not physical. Every point of that
+50% is bought from an assumption, which makes the constant **a dial that sets
the reported return** — a reason to keep it conservative, not to raise it.

**On making it adaptive**: the cap already IS proportional to volume
(`quote_volume / MULT`); what is fixed is the participation RATE. Raising that
rate on evidence would require order-book DEPTH, not candle volume — candle
volume is a poor proxy for what can be absorbed at the touch. freqtrade exposes
`check_depth_of_market` (currently disabled) for this.

**Recommendation: keep 10.** Both sweeps agree that the constant is not
load-bearing on trade quality, and there is no slippage model to constrain a
larger value. The evidence to justify a higher rate has to come from live or
dry-run fill data — requested size versus realised fill price — which is the
same dry run already required before this configuration goes live.
