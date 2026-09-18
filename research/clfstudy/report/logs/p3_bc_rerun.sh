#!/bin/bash
# Phase-3 window-B/C re-run (author decision A, 2026-09-18), gated on a passing real-data smoke test.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
until [ -f $LOG/BC_RERUN_P2P4_DONE ] && [ -f $LOG/E2E_READY ]; do sleep 120; done
echo "p3bc: smoke test on GPU ($(date))"
if python -m research.clfstudy.smoke --arch LSTM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_gpu.log 2>&1; then
  echo PASS > $LOG/SMOKE_GPU_PASS
else
  echo FAIL > $LOG/SMOKE_GPU_FAIL; echo "p3bc: SMOKE FAILED — not starting the re-run"; exit 1
fi
cp research/clfstudy/results/results.jsonl $LOG/results_before_p3_bc_rerun.jsonl
python - <<'PY'
import json
FIX='2026-09-18T07:00:00'
src='research/clfstudy/results/results.jsonl'; keep=[]; n=0
for l in open(src):
    r=json.loads(l)
    if r['phase']==3 and r['window_id'] in 'BC' and r['seed'] in (0,1) and r['started_at']<FIX: n+=1; continue
    keep.append(l)
open(src,'w').writelines(keep); print('dropped', n, 'contaminated phase-3 B/C rows')
PY
echo "p3bc: phase 3 B/C re-run ($(date))"
python -m research.clfstudy.phase --phase 3 >> $LOG/phase_3_bc_rerun.log 2>&1
python -m research.clfstudy.phase --phase 3 --retry-failed >> $LOG/phase_3_bc_rerun.log 2>&1
python -m research.clfstudy.leaderboard --phase 3 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 3 >> $LOG/gate_3_after_bc_rerun.log 2>&1
echo "p3bc: DONE ($(date))" | tee $LOG/P3_BC_RERUN_DONE
