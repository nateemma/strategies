#!/bin/bash
# Reviewer-4 cheap diagnostics: read-out (pool last vs mean) and patience 40, on the TCN/Transformer anchors, window C, 3 seeds.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/SIZE32_DONE ] && [ -f research/clfstudy/report/logs/POOL_TOGGLE_READY ]; do sleep 120; done
run() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"hp\":$2,\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":0,\"train\":$3,\"tags\":[\"pool-patience-diag\",\"$4\"]}" --window C --seed $5; }
for s in 0 1 2; do
  run TCN_FAM '{"channels":64,"depth":4,"dilation_base":2,"pool":"last"}' '{"lr":0.0003,"stride":2}' tcn_pool_last $s
  run TRANSFORMER_FAM '{"d_model":64,"layers":1,"pool":"last"}' '{"lr":0.0003,"stride":2}' tr_pool_last $s
  run TCN_FAM '{"channels":64,"depth":4,"dilation_base":2}' '{"lr":0.0003,"stride":2,"patience":40}' tcn_pat40 $s
  run TRANSFORMER_FAM '{"d_model":64,"layers":1}' '{"lr":0.0003,"stride":2,"patience":40}' tr_pat40 $s
done
echo DONE > research/clfstudy/report/logs/POOL_DIAG_DONE
