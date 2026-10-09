#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "chain4: third seeds ($(date))"
bash research/clfstudy/configs/phase_3_seed3.sh >> $LOG/phase_3_seed3.log 2>&1
echo "chain4: gate 3 re-evaluation ($(date))"
python -m research.clfstudy.gate --phase 3 >> $LOG/gate_3_after_seed3.log 2>&1
python -m research.clfstudy.leaderboard --phase 3 >> $LOG/leaderboard.log 2>&1
echo "chain4: phase 4 ($(date))"
python -m research.clfstudy.phase --phase 4 >> $LOG/phase_4.log 2>&1
echo "chain4: leaderboard 4 ($(date))"
python -m research.clfstudy.leaderboard --phase 4 >> $LOG/leaderboard.log 2>&1
echo "chain4: DONE ($(date))" | tee $LOG/CHAIN4_DONE
