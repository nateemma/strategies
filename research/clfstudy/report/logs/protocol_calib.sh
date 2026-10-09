#!/bin/bash
# Training-protocol calibration: production LSTM, window C, 3 seeds, full data (stride 1), phase 0 (excluded from leaderboards).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
run() { python -m research.clfstudy.run --config "{\"arch\":\"LSTM\",\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":$1,\"tags\":[\"protocol-calibration\",\"$2\"]}" --window C --seed $3; }
for s in 0 1 2; do run '{"lr":0.0003}' lr3e-4 $s; done
for s in 0 1 2; do run '{"patience":20}' pat20 $s; done
echo DONE > research/clfstudy/report/logs/PROTOCOL_CALIB_DONE
