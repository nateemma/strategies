#!/bin/bash
# GPU ablation of the pure-LSTM optimisation lottery (window C, 3 seeds per arm, stride 2, lr 3e-4 unless stated).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/CHAIN4C_DONE ] && [ -f research/clfstudy/report/logs/RNN_TOGGLES_READY ]; do sleep 120; done
run() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"hp\":$2,\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":$3,\"tags\":[\"rnn-ablation\",\"$4\"]}" --window C --seed $5; }
for s in 0 1 2; do
  run LSTM_FAM '{"hidden":64}' '{"lr":0.0003,"stride":2}' default $s
  run LSTM_FAM '{"hidden":64,"pool_norm":false}' '{"lr":0.0003,"stride":2}' no_poolnorm $s
  run LSTM_FAM '{"hidden":64,"input_proj":true}' '{"lr":0.0003,"stride":2}' input_proj $s
  run LSTM_FAM '{"hidden":64,"head":"bottleneck"}' '{"lr":0.0003,"stride":2}' bottleneck $s
  run LSTM_FAM '{"hidden":64,"input_proj":true,"head":"bottleneck"}' '{"lr":0.0003,"stride":2}' proj_bottleneck $s
  run LSTM_FAM '{"hidden":64}' '{"lr":0.0003,"stride":2,"batch_size":1024}' batch1024 $s
  run LSTM_FAM '{"hidden":64}' '{"lr":0.0001,"stride":2,"patience":20}' lr1e-4_pat20 $s
  run LSTM_FAM '{"hidden":64}' '{"lr":0.0003,"stride":2,"weight_decay":0.0001}' wd1e-4 $s
  run LSTM_RAW '{}' '{"lr":0.0003,"stride":2}' raw_control $s
done
echo DONE > research/clfstudy/report/logs/RNN_ABLATION_DONE
