#!/bin/bash
# Re-run the contaminated window-B/C rows (seeds 0-1) of phase 2 (RIDGE, ROCKET) and phase 4 non-RNN configs from scratch.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/TCN_CLONE_DONE ]; do sleep 120; done
cp research/clfstudy/results/results.jsonl research/clfstudy/report/logs/results_before_bc_rerun_p2p4.jsonl
python - <<'PY'
import json
ids=set("dbd824e7736c,a09a0b24d6fe,b85e94f80dc6,12af479dfaa3,f7d18cb3a30f,fa4e5ad070f7,318ea175aa05,486f44b1f862,67ffa1f12fbd,d5a1481bd4c0,9278e1441b47,821f46c115ab,bbbed50c3b5c,6fe7246246f5,a4fb5eb48337,3413d77700b0,e5974774aa5f,8e1f1550ad56,9c94bdef2a2d,1260944a8d42,4a28d1ce4ebd,8a9bbc798cf3,b8285305a779".split(',')) | set("d1dc0339b10b,12c3636e7110".split(','))
FIX='2026-09-18T07:00:00'
src='research/clfstudy/results/results.jsonl'; keep=[]; n=0
for l in open(src):
    r=json.loads(l)
    if r['config_id'] in ids and r['window_id'] in 'BC' and r['seed'] in (0,1) and r['phase'] in (2,4) and r['started_at']<FIX: n+=1; continue
    keep.append(l)
open(src,'w').writelines(keep); print('dropped', n, 'contaminated B/C rows (phases 2 and 4)')
PY
python -m research.clfstudy.phase --phase 2 >> research/clfstudy/report/logs/phase_2_bc_rerun.log 2>&1
python -m research.clfstudy.phase --phase 4 --only dbd824e7736c,a09a0b24d6fe,b85e94f80dc6,12af479dfaa3,f7d18cb3a30f,fa4e5ad070f7,318ea175aa05,486f44b1f862,67ffa1f12fbd,d5a1481bd4c0,9278e1441b47,821f46c115ab,bbbed50c3b5c,6fe7246246f5,a4fb5eb48337,3413d77700b0,e5974774aa5f,8e1f1550ad56,9c94bdef2a2d,1260944a8d42,4a28d1ce4ebd,8a9bbc798cf3,b8285305a779 >> research/clfstudy/report/logs/phase_4_bc_rerun.log 2>&1
python -m research.clfstudy.leaderboard --phase 2 >> research/clfstudy/report/logs/leaderboard.log 2>&1
python -m research.clfstudy.leaderboard --phase 4 >> research/clfstudy/report/logs/leaderboard.log 2>&1
echo DONE > research/clfstudy/report/logs/BC_RERUN_P2P4_DONE
