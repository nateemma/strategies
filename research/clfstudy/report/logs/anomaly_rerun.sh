#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/BASELINE_SEED2_DONE ]; do sleep 300; done
python -m research.clfstudy.run --config '{"arch":"TRANSFORMER_FAM","hp":{"d_model":64,"pos":"rope"},"input_set":"ind24","seq_len":16,"split_set":"v1","phase":4,"train":{"lr":0.0003,"stride":2},"tags":["transformer","pos"]}' --window B --seed 1
python -m research.clfstudy.leaderboard --phase 4 >> research/clfstudy/report/logs/leaderboard.log 2>&1
echo DONE > research/clfstudy/report/logs/ANOMALY_DONE
