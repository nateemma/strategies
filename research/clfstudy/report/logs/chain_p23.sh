#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
python -m research.clfstudy.smoke --arch MLP_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_p23.log 2>&1 || { echo FAIL > $LOG/SMOKE23_FAIL; exit 1; }
python -m research.clfstudy.phase --phase 23 >> $LOG/phase_23.log 2>&1
python -m research.clfstudy.phase --phase 23 --retry-failed >> $LOG/phase_23.log 2>&1
python -m research.clfstudy.leaderboard --phase 23 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.screen --phase 23 --baseline-phase 19 > $LOG/screen_23.log 2>&1
echo "p23: DONE ($(date))" | tee $LOG/P23_DONE
