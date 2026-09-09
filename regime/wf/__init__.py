"""Pooled walk-forward evaluation harness (feature 007).

A MEASUREMENT INSTRUMENT. Its deliverable is statistical power, not P&L. Nothing here may alter an
entry gate, an exit parameter, a label, or a model configuration (FR-015).

The baseline constants below were MEASURED on 2026-09-09 by bootstrapping NNNC_MLX's own W1
per-trade P&L (99 trades, +9.831%, archive backtest-result-2026-09-08_19-55-22.zip). They are
pinned, not recomputed (research R1).
"""

BASELINE_HALF_WIDTH_PP = 7.633      # 95% CI half-width, single window, unpaired
BASELINE_PAIRED_PP = 6.1            # half-width when arms are paired on their symmetric difference
BASELINE_N_TRADES = 99
BASELINE_RETURN_PCT = 9.831
BASELINE_CI = (2.383, 17.649)
BASELINE_SD_PP = 3.893

BOOTSTRAP_SEED = 0                  # must match regime/tail/compare.py (FR-022)
N_RESAMPLES = 20000

# Honest ceiling (research R2): pooling every fold the data allows, at double the universe, moves
# the floor to ~2.5-2.8pp — a ~2.7x improvement, NOT 10x. Effects of 3-7pp become newly
# measurable; 1-2pp parameter effects remain unresolvable AFTER this ships.
TARGET_HALF_WIDTH_PP = 4.0          # SC-001a — a TARGET, missable; missing it is a RESULT
TARGET_POOLED_TRADES = 350          # SC-002
