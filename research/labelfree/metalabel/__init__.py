"""Meta-labelling from the pooled walk-forward trade archive (feature 011, brief 03).

A secondary model trained on the REALISED outcomes of trades the primary strategy actually
took, using only what was knowable at each trade's entry bar. The primary model, its guards,
its exits and its stop are not touched.

The constants below were MEASURED from `evaluation/pooled_nnnc.json` on 2026-09-12. They are
pinned, not chosen. `tests/test_metalabel.py` asserts the archive still matches them: if any
has moved, the floor and the base rate must be recomputed before anything else runs.
"""
from __future__ import annotations

EVAL_FOLDS = tuple(f"f{i:02d}" for i in range(5, 15))   # folds 5-14, the evaluation span
WARMUP_FOLDS = tuple(f"f{i:02d}" for i in range(0, 5))  # folds 0-4, too few trades to evaluate

N_ARCHIVE_TRADES = 378        # pooled archive, all 15 folds
N_EVAL_TRADES = 287           # folds 5-14
N_WARMUP_TRADES = 91          # folds 0-4 — also f05's entire training set
EVAL_YEARS = 2.45             # 895 test days / 365

# SC-009 kill gate: nested OOS discrimination below this for every threshold kills the feature.
AUC_KILL = 0.55

# Base win rate on the evaluation span (210/287). A model that accepts everything scores this,
# which is why ACCURACY IS NEVER REPORTED — it cannot distinguish skill from the majority class.
BASE_WIN_RATE = 0.732

# Recomputed floor for the evaluation span (research R2). NOT the study-wide 3.350 pp/yr:
# that was measured on 378 trades over 3.68 years. Bootstrapping the same way the harness does
# (20,000 resamples, seed 0) over the 287 evaluation trades gives a half-width of 11.479 pp
# across 2.45 years. The floor here is 40% HIGHER, and a delta between the two figures clears
# the old floor while being unresolvable on this span.
FLOOR_PP_PER_YEAR = 4.681

BOOTSTRAP_SEED = 0            # must match evaluation/ and research/labelfree/stats.py
N_RESAMPLES = 20000

ARCHIVE_PATH = "evaluation/pooled_nnnc.json"
MODEL_STORE = "saved_data"    # per-fold primary models: saved_data/NNNC_WF_f{NN}/


# ---------------------------------------------------------------------------
# THE TRADE TABLE (T021) — `rows.jsonl`, alongside `rows_summary.json`.
#
# Reusable by any brief that needs entry-time state for real trades. Briefs 5 and 8 were
# named as consumers; nothing here is specific to the veto model.
#
# One JSON object per line, one line per archived trade, 378 of them:
#
#   fold_id            f00..f14
#   pair, open_date    the join key; unique across the table
#   close_date
#   profit_abs         REALISED money under real exits, fees and fills
#   profit_ratio, exit_reason, scale_pp_per_abs
#   target_win         profit_abs > 0
#   features           dict, described below
#
# FEATURES, all valued at the ENTRY BAR (`open_date - 15min`):
#
#   %predict_prob_sell   the primary model's own class probabilities, recovered from the
#   %predict_prob_hold   framework side-car (FT_PREDICT_PROB_SINK). NOT available from
#   %predict_prob_buy    freqtrade's --export signals: that artefact is built before the
#                        entry/exit phase runs. See research R11.
#   guard_metric, rvol, atr_pct, bb_width, adx, volume, close
#                        market state, from the signals export
#
# DELIBERATELY ABSENT: pair identity, pair group, and any statistic derived from a pair's
# own trade history. They are causal and would pass the leakage audit, but they hand a
# model a per-pair quality score, and the cheapest route to discrimination then becomes
# ranking pairs — which is a finding this repo has already recorded and closed. Without
# them, any measured gain is necessarily WITHIN-pair.
#
# GUARANTEES, each measured rather than asserted:
#   - every fold's re-inferred (pair, open_date) set equals the archive's EXACTLY
#   - no feature changes when bars after the entry are deleted (bit-identical, RTOL 0)
#   - the leakage audit detects a planted leak it is required to catch
#
# CAVEAT: the probability columns are NaN wherever the model did not score a bar (the
# front-padded warm-up region). Such rows are excluded and counted, never imputed, and
# never zero-filled — 0.0 is a legitimate probability meaning "confidently not this class".
# ---------------------------------------------------------------------------
