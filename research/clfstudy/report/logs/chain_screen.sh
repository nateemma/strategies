#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "screen: smoke ($(date))"
python -m research.clfstudy.smoke --arch LSTM_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_screen.log 2>&1 || { echo FAIL > $LOG/SMOKE_SCREEN_FAIL; exit 1; }
for P in 20 21; do
  echo "screen: phase $P ($(date))"
  python -m research.clfstudy.phase --phase $P >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.phase --phase $P --retry-failed >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.leaderboard --phase $P >> $LOG/leaderboard.log 2>&1
  python -m research.clfstudy.screen --phase $P --baseline-phase 19 > $LOG/screen_$P.log 2>&1
done
echo "screen: building phase 22 ($(date))"
python $LOG/make_phase22.py >> $LOG/screen_22.log 2>&1
python -m research.clfstudy.phase --phase 22 >> $LOG/phase_22.log 2>&1
python -m research.clfstudy.phase --phase 22 --retry-failed >> $LOG/phase_22.log 2>&1
python -m research.clfstudy.leaderboard --phase 22 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.screen --phase 22 --baseline-phase 19 >> $LOG/screen_22.log 2>&1
echo "screen: DONE ($(date))" | tee $LOG/SCREEN_DONE
