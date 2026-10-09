#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "chain4b: phase 4 lr3e-4 stride2 ($(date))"
python -m research.clfstudy.phase --phase 4 >> $LOG/phase_4b.log 2>&1
echo "chain4b: leaderboard 4 ($(date))"
python -m research.clfstudy.leaderboard --phase 4 >> $LOG/leaderboard.log 2>&1
echo "chain4b: DONE ($(date))" | tee $LOG/CHAIN4B_DONE
