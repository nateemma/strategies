#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
python -m research.clfstudy.smoke --arch LSTM_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_abc.log 2>&1 || { echo FAIL > $LOG/SMOKE_ABC_FAIL; exit 1; }
for P in 26 27 28; do
  python -m research.clfstudy.phase --phase $P >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.phase --phase $P --retry-failed >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.leaderboard --phase $P >> $LOG/leaderboard.log 2>&1
  echo "p$P: DONE ($(date))" | tee $LOG/P${P}_DONE
done
python -m research.clfstudy.screen --phase 26 --baseline-phase 19 > $LOG/screen_26.log 2>&1
echo "abc: DONE ($(date))" | tee $LOG/ABC_DONE
