#!/bin/bash
# Phase 31: LSTM h32 d2 at window 128 on forward_mae, WINDOW A ONLY (2 runs).
# --max-runs 2 == window A seeds 0,1 (windows iterate A,B,C outer; seeds inner).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
python -m research.clfstudy.phase --phase 31 --max-runs 2 >> $LOG/phase_31.log 2>&1
echo "p31 (window A probe): DONE ($(date))" | tee $LOG/P31_A_DONE
