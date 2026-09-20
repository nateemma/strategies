#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
until [ -f $LOG/P25_DONE ]; do sleep 120; done
python -m research.clfstudy.phase --phase 24 >> $LOG/phase_24.log 2>&1
python -m research.clfstudy.phase --phase 24 --retry-failed >> $LOG/phase_24.log 2>&1
python -m research.clfstudy.leaderboard --phase 24 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.screen --phase 24 --baseline-phase 19 > $LOG/screen_24.log 2>&1
echo "p24: DONE ($(date))" | tee $LOG/P24_DONE
