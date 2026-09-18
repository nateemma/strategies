#!/bin/bash
# Re-run the phase-4 RNN family configs from scratch under the fresh-start harness (all earlier RNN rows predate the fix).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/CLEAN_RERUN_DONE ]; do sleep 120; done
cp research/clfstudy/results/results.jsonl research/clfstudy/report/logs/results_before_rnn_sweep_rerun.jsonl
python - <<'PY'
import json
ids=set("5616ab5d0bd0,7b50342154ef,b5b58a6c2cc5,b690297cbbf5,f5be6cc3a0df,702b00fcb20c,70eb48f9acfb,68c1f710246f,cef2eef58d6b,3c120e544d4e,4c265676afae,551b8bb6dcf2".split(','))
src='research/clfstudy/results/results.jsonl'; keep=[]; n=0
for l in open(src):
    r=json.loads(l)
    if r['phase']==4 and r['config_id'] in ids: n+=1; continue
    keep.append(l)
open(src,'w').writelines(keep); print('dropped', n, 'stale RNN phase-4 rows')
PY
for i in $(echo "5616ab5d0bd0,7b50342154ef,b5b58a6c2cc5,b690297cbbf5,f5be6cc3a0df,702b00fcb20c,70eb48f9acfb,68c1f710246f,cef2eef58d6b,3c120e544d4e,4c265676afae,551b8bb6dcf2" | tr ',' ' '); do rm -f research/clfstudy/.cache/ckpt/${i}_*.safetensors; done
python -m research.clfstudy.phase --phase 4 --only 5616ab5d0bd0,7b50342154ef,b5b58a6c2cc5,b690297cbbf5,f5be6cc3a0df,702b00fcb20c,70eb48f9acfb,68c1f710246f,cef2eef58d6b,3c120e544d4e,4c265676afae,551b8bb6dcf2 >> research/clfstudy/report/logs/phase_4_rnn.log 2>&1
python -m research.clfstudy.leaderboard --phase 4 >> research/clfstudy/report/logs/leaderboard.log 2>&1
echo DONE > research/clfstudy/report/logs/RNN_SWEEP_DONE
