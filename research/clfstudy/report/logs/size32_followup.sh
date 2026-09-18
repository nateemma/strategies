#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/TCN_ABLATION_DONE ]; do sleep 120; done
python -m research.clfstudy.phase --phase 4 --only 8a9bbc798cf3,b8285305a779 >> research/clfstudy/report/logs/phase_4d.log 2>&1
echo DONE > research/clfstudy/report/logs/SIZE32_DONE
