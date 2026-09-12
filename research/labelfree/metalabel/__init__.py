"""Meta-labelling from the pooled walk-forward trade archive (feature 011, brief 03).

A secondary model trained on the REALISED outcomes of trades the primary strategy actually
took, using only what was knowable at each trade's entry bar. The primary model, its guards,
its exits and its stop are not touched.

The constants below were MEASURED from `regime/wf/pooled_nnnc.json` on 2026-09-12. They are
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

BOOTSTRAP_SEED = 0            # must match regime/wf and research/labelfree/stats.py
N_RESAMPLES = 20000

ARCHIVE_PATH = "regime/wf/pooled_nnnc.json"
MODEL_STORE = "saved_data"    # per-fold primary models: saved_data/NNNC_WF_f{NN}/
