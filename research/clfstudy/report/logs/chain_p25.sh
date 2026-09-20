#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
python -m research.clfstudy.smoke --arch TCN_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_p25.log 2>&1 || { echo FAIL > $LOG/SMOKE25_FAIL; exit 1; }
python -m research.clfstudy.phase --phase 25 >> $LOG/phase_25.log 2>&1
python -m research.clfstudy.phase --phase 25 --retry-failed >> $LOG/phase_25.log 2>&1
python -m research.clfstudy.leaderboard --phase 25 >> $LOG/leaderboard.log 2>&1
echo "p25: DONE ($(date))" | tee $LOG/P25_DONE
