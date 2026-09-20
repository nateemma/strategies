#!/bin/bash
# Follow-on to chain_abc.sh: waits for ABC_DONE, then runs phase 29 (the phase-25
# architecture x window-length mirror on the forward_mae dd10 label).
# Deliberately a SEPARATE script: chain_abc.sh is mid-execution and bash reads a
# running script incrementally by byte offset, so editing it in place would corrupt it.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs

# Only start once the abc chain has finished (or has died).
while [ ! -f $LOG/ABC_DONE ]; do
  pgrep -f "chain_abc.sh" >/dev/null || { echo "abc chain gone without ABC_DONE" > $LOG/P29_ABORT; exit 1; }
  sleep 120
done

python -m research.clfstudy.smoke --arch LSTM_FAM --stride 16 --epochs 3 --pairs LINK/USDT \
  > $LOG/smoke_p29.log 2>&1 || { echo FAIL > $LOG/SMOKE_P29_FAIL; exit 1; }

python -m research.clfstudy.phase --phase 29 >> $LOG/phase_29.log 2>&1
python -m research.clfstudy.phase --phase 29 --retry-failed >> $LOG/phase_29.log 2>&1
python -m research.clfstudy.leaderboard --phase 29 >> $LOG/leaderboard.log 2>&1
echo "p29: DONE ($(date))" | tee $LOG/P29_DONE
