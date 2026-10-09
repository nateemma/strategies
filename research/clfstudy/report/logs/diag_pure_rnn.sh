#!/bin/bash
# Discriminating test: existing pure-ish LSTM types under the amended protocol (lr 3e-4, stride 2), window C, seed 0.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
run() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"hp\":$2,\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":{\"lr\":0.0003,\"stride\":2},\"tags\":[\"diag-pure-rnn\",\"$3\"]}" --window C --seed 0; }
run LSTM_RAW '{}' raw
run LSTM_NOCONV '{}' noconv
run GRU '{}' gru_prod
echo DONE > research/clfstudy/report/logs/DIAG_PURE_RNN_DONE
