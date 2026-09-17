# Quickstart: validate the study harness end to end

Prerequisites: `cd /Users/philprice95/projects/freqtrade/user_data/strategies`,
`source ../../.venv/bin/activate`, `export PYTHONPATH=.`, binanceus 15m feathers present for
the 11 pairs.

```bash
# 1. unit tests (seconds)
pytest research/clfstudy/tests -q

# 2. label fidelity: shim == StrategyResolver-loaded NNNC_MLX (one pair, ~1 min)
python scripts/check_label_fidelity.py --pair LINK/USDT

# 3. split set
python -m research.clfstudy.splits build --set v1

# 4. calibration: minutes per run, projected totals
python -m research.clfstudy.calibrate --arch LSTM

# 5. phase 2 baselines then phase 3 existing types (overnight)
python -m research.clfstudy.phase --phase 2
python -m research.clfstudy.phase --phase 3
python -m research.clfstudy.leaderboard --phase 3      # ⏸ checkpoint: first results table

# 6. gate, then later phases, each followed by leaderboard + gate + checkpoint
python -m research.clfstudy.gate --phase 3
python -m research.clfstudy.phase --phase 4 && python -m research.clfstudy.leaderboard --phase 4 && python -m research.clfstudy.gate --phase 4
# ... phases 5, 6, 7 likewise

# 7. importance on the best model (saved with --save-weights in phase 7), then the report
python -m research.clfstudy.importance --config-id <best> --window A
python -m research.clfstudy.report
```

Expected outcomes: step 1 green; step 2 prints `IDENTICAL (n=…)`; step 3 prints three windows
with A excluding XRP and SUI; step 5's table shows MAJORITY, PERSISTENCE, RIDGE, ROCKET and
the 24 existing types with per-window and pooled mean ± std; every row has `epochs_run ≤ 40`.
