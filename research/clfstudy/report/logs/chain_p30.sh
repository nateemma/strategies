#!/bin/bash
# Follow-on to chain_p29.sh: waits for P29_DONE, then runs phase 30 (LSTM at window 256).
# Separate script for the same reason as chain_p29.sh: bash reads a running script
# incrementally by byte offset, so editing a live chain in place would corrupt it.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs

while [ ! -f $LOG/P29_DONE ]; do
  pgrep -f "chain_p29.sh" >/dev/null || { echo "p29 chain gone without P29_DONE" > $LOG/P30_ABORT; exit 1; }
  sleep 300
done

python -m research.clfstudy.phase --phase 30 >> $LOG/phase_30.log 2>&1
python -m research.clfstudy.phase --phase 30 --retry-failed >> $LOG/phase_30.log 2>&1
python -m research.clfstudy.leaderboard --phase 30 >> $LOG/leaderboard.log 2>&1
echo "p30: DONE ($(date))" | tee $LOG/P30_DONE
