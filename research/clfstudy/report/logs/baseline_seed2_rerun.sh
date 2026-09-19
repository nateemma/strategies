#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/FILL_DONE ]; do sleep 300; done
python -m research.clfstudy.phase --phase 2 >> research/clfstudy/report/logs/phase_2_seed2_rerun.log 2>&1
python -m research.clfstudy.leaderboard --phase 2 >> research/clfstudy/report/logs/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 4 >> research/clfstudy/report/logs/gate_4_final.log 2>&1
echo DONE > research/clfstudy/report/logs/BASELINE_SEED2_DONE
