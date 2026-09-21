#!/bin/bash
# Follow-on to chain_p29.sh: waits for P29_DONE, then runs phase 30 (LSTM at window 256)
# on EVALUATION WINDOW A ONLY -- a 2-run direction probe, not the full ladder rung.
#
# --max-runs 2 is exactly window A seeds 0,1: phase.expand_runs iterates windows in
# split-set order (A, B, C) outer and seeds inner, and v1's first window is A
# (test 2023-07-01..2023-12-31). No code change needed, and because the rows key on
# config_id, running B and C later just resumes -- window A will be skipped.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs

while [ ! -f $LOG/P29_DONE ]; do
  pgrep -f "chain_p29.sh" >/dev/null || { echo "p29 chain gone without P29_DONE" > $LOG/P30_ABORT; exit 1; }
  sleep 300
done

python -m research.clfstudy.phase --phase 30 --max-runs 2 >> $LOG/phase_30.log 2>&1
echo "p30 (window A probe): DONE ($(date))" | tee $LOG/P30_A_DONE
