#!/bin/bash
# Clean re-run of keys that warm-started from stale checkpoints (see methods.md). Runs after the diagnostics queue and after run.py's fresh-start fix.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/POOL_DIAG_DONE ] && [ -f research/clfstudy/report/logs/FRESH_START_READY ]; do sleep 120; done
cp research/clfstudy/results/results.jsonl research/clfstudy/report/logs/results_before_clean_rerun.jsonl
python - <<'PY'
import json
drop={('098ca2f95a2e','A',0),('076666f04306','A',1),('666675f56e77','B',1)} | {('0b31c7bb9404',w,s) for w in 'ABC' for s in (0,1)}
src='research/clfstudy/results/results.jsonl'; keep=[]; n=0
for l in open(src):
    r=json.loads(l)
    if (r['config_id'],r['window_id'],r['seed']) in drop and r['phase']==3: n+=1; continue
    keep.append(l)
open(src,'w').writelines(keep); print('dropped', n, 'rows for clean re-run')
PY
rm -f research/clfstudy/.cache/ckpt/{098ca2f95a2e_0,076666f04306_1,666675f56e77_1,0b31c7bb9404_0,0b31c7bb9404_1}.safetensors
run3() { python -m research.clfstudy.run --config "{\"arch\":\"$1\",\"input_set\":\"ind24\",\"seq_len\":16,\"split_set\":\"v1\",\"phase\":3,\"tags\":[\"clean-rerun\"]}" --window $2 --seed $3; }
run3 LSTM2 A 0; run3 CNN A 1; run3 Mamba2 B 1
for w in A B C; do for s in 0 1; do run3 Multihead $w $s; done; done
# RNN-ablation default arm, seeds 0 and 1 (seed 2 ran clean)
for s in 0 1; do python -m research.clfstudy.run --config '{"arch":"LSTM_FAM","hp":{"hidden":64},"input_set":"ind24","seq_len":16,"split_set":"v1","phase":0,"train":{"lr":0.0003,"stride":2},"tags":["rnn-ablation","default"]}' --window C --seed $s; done
python -m research.clfstudy.leaderboard --phase 3 >> research/clfstudy/report/logs/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 3 >> research/clfstudy/report/logs/gate_3_after_clean.log 2>&1
echo DONE > research/clfstudy/report/logs/CLEAN_RERUN_DONE
