#!/bin/bash
# Third seed (seed 2) on the failing window(s) only, for the 12 stability-only failures of gate 3.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
run() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":3,\"tags\":[\"third-seed\"]}" --window "$2" --seed 2; }
run LSTM C; run Wavenet C; run LSTM_NORES C; run LSTM_NOCONV C; run LSTM_D10 C; run LSTM_NOISY C
run LSTM_RAW C; run LSTM_INJIT C; run Mamba2 C; run LSTM_NORESIZE B; run LSTM_NORESIZE C
run LSTM_PONDER B; run LSTM_POOL B
