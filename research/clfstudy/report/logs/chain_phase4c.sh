#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "chain4c: phase 4 non-RNN configs ($(date))"
python -m research.clfstudy.phase --phase 4 --only dbd824e7736c,a09a0b24d6fe,b85e94f80dc6,12af479dfaa3,f7d18cb3a30f,fa4e5ad070f7,318ea175aa05,486f44b1f862,67ffa1f12fbd,d5a1481bd4c0,9278e1441b47,821f46c115ab,bbbed50c3b5c,6fe7246246f5,a4fb5eb48337,3413d77700b0,e5974774aa5f,8e1f1550ad56,9c94bdef2a2d,1260944a8d42,4a28d1ce4ebd >> $LOG/phase_4c.log 2>&1
echo "chain4c: DONE ($(date))" | tee $LOG/CHAIN4C_DONE
