#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/RNN_ABLATION_DONE ]; do sleep 120; done
run() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"hp\":$2,\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":{\"lr\":0.0003,\"stride\":2},\"tags\":[\"tcn-ablation\",\"$3\"]}" --window C --seed $4; }
for s in 0 1 2; do
  run TCN_FAM '{"channels":64,"depth":8,"dilation_base":2}' depth8_dil2 $s
  run Wavenet '{}' wavenet_control $s
done
echo DONE > research/clfstudy/report/logs/TCN_ABLATION_DONE
