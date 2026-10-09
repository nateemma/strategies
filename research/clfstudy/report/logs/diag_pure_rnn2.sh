#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/DIAG_PURE_RNN_DONE ]; do sleep 30; done
run() { python -m research.clfstudy.run --config "{\"arch\":\"LSTM_FAM\",\"hp\":$1,\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":{\"lr\":0.0003,\"stride\":2},\"tags\":[\"diag-pure-rnn\",\"$2\"]}" --window C --seed 0; }
run '{"hidden":64,"input_norm":false}' fam_noinputnorm
run '{"hidden":64,"input_norm":false,"pool_norm":false}' fam_nonorm
echo DONE > research/clfstudy/report/logs/DIAG_PURE_RNN2_DONE
