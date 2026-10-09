#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
python -m research.clfstudy.phase --phase 3 --only 0b31c7bb9404 >> research/clfstudy/report/logs/multihead_rerun.log 2>&1
python -m research.clfstudy.leaderboard --phase 3 >> research/clfstudy/report/logs/leaderboard.log 2>&1
echo "DONE $(date)" > research/clfstudy/report/logs/MULTIHEAD_DONE
