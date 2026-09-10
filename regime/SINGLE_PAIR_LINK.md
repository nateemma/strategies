# Single-pair (LINK) variant — feature 008, measured 2026-09-10

**Answer: cross-pair training contributes essentially nothing to LINK selection. 38 of 39 trades
are identical.** A model that has never seen the other ten pairs picks the same LINK trades as one
trained on all eleven.

## Result

| window | variant | control | shared | identical | differs |
|---|---|---|---|---|---|
| **W1** (OOS) | 8 trades | 8 trades | 8 | **8** | 0 |
| W2 | 26 | 26 | 26 | **25** | 1 |
| W3 | 5 | 5 | 5 | **5** | 0 |

Per-trade expectancy is identical wherever selection is identical:

| window | variant | control |
|---|---|---|
| W1 | +2.270% [+1.289, +3.327] | +2.270% [+1.289, +3.327] |
| W2 | +0.355% [−1.165, +1.685] | +0.454% [−1.050, +1.763] |
| W3 | +2.500% [+1.644, +3.356] | +2.500% [+1.644, +3.356] |

LINK's contribution to whole-portfolio return: W1 +1.143pp both arms (delta 0.000);
W2 +0.805 vs +1.036 (delta −0.231pp, ONE differing trade); W3 +1.113pp both (delta 0.000).

**Every delta is UNRESOLVED** and could not have been otherwise: this pair supplies 8 trades on W1
against a noise floor of 7.633pp measured on 99, and the comparison needs 20 differing trades
before a bootstrap can even estimate a distribution. That is why the feature was built around
selection and expectancy rather than returns.

## What it means

The gate, not the model, is choosing these trades. The two models were trained on entirely
different data — one pair versus eleven — and converged on the same LINK entries. That is the
strongest form yet of the standing finding that the entry gate dominates model choice: it survives
changing the *training set* by an order of magnitude, not just changing hyperparameters.

Practically: **cross-pair training is neither helping nor hurting LINK.** There is no dilution to
remove, and no single-pair specialisation to gain.

## The finding that outlived the feature: DOMINANCE ROTATES

Measured while scoping this, and independent of anything the variant showed:

| pair | W1 %P&L | W2 %P&L | W3 %P&L |
|---|---|---|---|
| ZEC | **44.9%** | 7.0% | 2.8% |
| XRP | 23.6% | **38.3%** | −3.1% |
| NEAR | 5.0% | 6.9% | **27.5%** |
| LINK | 11.6% | 9.5% | 26.4% |
| SUI | 4.0% | 21.6% | 3.2% |
| AVAX | — | −2.8% | 21.9% |

Top-3 by window: W1 ZEC/XRP/LINK · W2 XRP/SUI/DOT · W3 NEAR/LINK/AVAX.

**"The edge is ZEC" is a W1 observation, not a property of the strategy.** ZEC is 45% of W1 and
2.8% of W3. The feature began as ZEC-only and was re-scoped to LINK for exactly this reason — LINK
is the only pair contributing materially in all three windows. Any future single-pair or
pair-selection work should start here: there is no stable dominant pair to select.

## Constitution note (Principle X)

A LINK-only *win* would NOT have licensed shipping a per-pair model — Principle X prohibits
per-pair overrides, and the structural response to dilution would be a change to how ONE model
trains, not eleven models. The point is moot: there was no win, and no loss.

## Scope

Pooled walk-forward evaluation (US4) was **SKIPPED** and recorded as skipped. Its gate required the
trade sets to DIFFER; they do not. With 38 of 39 trades identical there is nothing for pooled folds
to resolve, so FR-013, FR-014, FR-020 and SC-006 are vacuously satisfied per the spec's scope note.

## Reproducing

```
PATH="$PWD/.venv/bin:$PATH" zsh user_data/strategies/scripts/test_strat.sh \
  -c user_data/strategies/config/config_link.json -f -t 20220101-20250831 NNNC NNNC_MLX_LINK
... run_arm.py --strategy NNNC_MLX_LINK --window W1 --tag link_only \
      --config user_data/strategies/config/config_link.json
... regime/pair_restricted.py --arm link_only --control nnnc_v2 --pair LINK/USDT
```
