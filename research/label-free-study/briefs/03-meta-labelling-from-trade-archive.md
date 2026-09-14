# Brief 03 — Meta-labelling from the repo's own trade archive

**Family:** meta-labelling (López de Prado; Hudson & Thames 2022)
**Type:** semi-supervised — the "labels" are realised outcomes of trades the production
strategy actually took, not lookahead labels on bars
**Priority:** 3 (run in parallel with brief 1; independent code) · **Budget:** 1–2 sessions

## Hypothesis

The primary model (`NNNC_MLX` + guards + ATR stop) already selects ~100 trades/yr with a
positive but concentrated edge (≈70% of P&L from ZEC-like pairs, `us_spot_market_study.md`).
A **secondary model trained on the realised `profit_abs` of those trades** — features
available at entry only — can learn *when to trust* the primary signal and *how much* to
size it, raising net return without touching the primary model. This is label-free in the
sense that matters here: the target is money the strategy actually made or lost under the
real exit logic, fees and fills, not a forward-excursion stamp.

## Prior evidence

Hudson & Thames E-mini study: OOS Sharpe 0.18 → 0.40 / 0.26 → 0.32; validation gains far
larger than OOS; "if the primary model is bad, meta-labelling only reduces the downside."
Grade medium-weak but the mechanism is sound and the cost is trivial.

## Why it fits this repo unusually well

- `user_data/backtest_results/*.zip` holds `strategy.<Name>.trades[]` with
  `pair, open_date, close_date, profit_abs, exit_reason` for every arm ever run.
- The pooled walk-forward harness already produces **378 OOS trades** across 15 folds —
  that is the training set, and it is *out-of-sample by construction* for the primary model.
- `regime/breakout_tail.py`, `regime/liquidity.py` and `regime/instruments.py` already
  compute the candidate entry-time features (rvol, spread proxy, breadth, correlation).

## Design

- **Training rows:** one per primary trade from the pooled WF run. Features: the primary
  model's confidence, the guard metrics at entry, ATR/rvol, pair liquidity proxies
  (candle quote volume vs stake), time-in-fold, BTC context, and the pair's recent hit rate
  (causal). **No feature may use anything after `open_date`.**
- **Target:** `profit_abs > 0` (classification) and `profit_abs` (regression for sizing).
- **Model:** LightGBM / sklearn gradient boosting (the `Sklearn/` family already wraps
  these). No neural net — n ≈ 400.
- **Nested walk-forward:** the secondary model for fold *k* is trained only on primary trades
  from folds `< k`. Folds 0–4 are warm-up (too few trades); evaluate on folds 5–14.
- **Actions:** (a) veto — skip the trade if P(win) < τ; (b) size — stake ∝ predicted
  expectancy, capped by liquidity. Test veto first (fewer degrees of freedom).
- **Freqtrade integration:** `confirm_trade_entry()` in a thin `NNNC_MLX` subclass loads the
  secondary model — the primary model path is pinned per AGENT_GUIDE so nothing retrains.

## Experiments

1. Feature leakage audit: shuffle-test each feature (permute across trades; AUC must not
   move if the feature is causal-but-uninformative, and must not be suspiciously high).
2. Veto model, τ ∈ {0.4, 0.5, 0.6}, nested WF folds 5–14, 5 seeds of the booster.
3. Sizing model on the surviving trades.
4. Pair-concentration check: does the veto simply remove non-ZEC trades (which would be the
   "implicit pair selector" finding again), or does it discriminate *within* pairs?

## Acceptance / kill

Protocol §5, computed on folds 5–14 only (fewer trades → quote the floor recomputed for that
span). Additional:
- If AUC on nested OOS < 0.55 for every τ → kill; the entry-time features carry no
  information about outcome beyond what the primary already used.
- If the veto's gain disappears when ZEC is excluded → record as "pair selection", not
  meta-labelling, and stop.

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-meta-labelling/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/03-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-meta-labelling. This is research/label-free-study brief 03; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md and
briefs/03-meta-labelling-from-trade-archive.md. Build research/labelfree/metalabel/ that
(1) extracts entry-time features and realised outcomes for every trade in the pooled
walk-forward archive for NNNC_MLX (evaluation/, regime/arms/), (2) runs the leakage audit,
(3) trains the nested walk-forward veto model, and (4) wires the accepted model into a thin
NNNC_MLX subclass via confirm_trade_entry with the primary model path pinned. Never let a
feature read data after the trade's open_date. Log trials to research/labelfree/trials.jsonl.
Write briefs/results/03-<date>.md per protocol §6.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
