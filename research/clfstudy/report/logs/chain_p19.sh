#!/bin/bash
# 019: strict triple-barrier label swap. Smoke-gated; runs after the 018 tail jobs.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
until [ -f $LOG/ANOMALY_DONE ] && [ -f $LOG/GATE19_READY ]; do sleep 120; done
echo "p19: smoke ($(date))"
if ! python -m research.clfstudy.smoke --arch LSTM_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_p19.log 2>&1; then echo FAIL > $LOG/SMOKE19_FAIL; echo "p19: SMOKE FAILED"; exit 1; fi
echo "p19: phase 19 ($(date))"
python -m research.clfstudy.phase --phase 19 >> $LOG/phase_19.log 2>&1
python -m research.clfstudy.phase --phase 19 --retry-failed >> $LOG/phase_19.log 2>&1
python -m research.clfstudy.leaderboard --phase 19 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 19 --ridge-phase 19 >> $LOG/gate_19.log 2>&1
echo "p19: DONE ($(date))" | tee $LOG/P19_DONE
