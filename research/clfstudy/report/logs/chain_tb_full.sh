#!/bin/bash
# After the pt/sl shim fix: purge the +3%/-2% rows, re-run phase 19 at a true +/-2%, then the screening campaign (phases 20-22).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
until [ -f $LOG/SHIM_FIX_READY ]; do sleep 60; done
cp research/clfstudy/results/results.jsonl $LOG/results_before_tb_floor_fix.jsonl
python - <<'PY'
import json, glob, os
src='research/clfstudy/results/results.jsonl'; keep=[]; n=0; ids=set()
for l in open(src):
    r=json.loads(l)
    if r['phase'] in (19,20,21,22): n+=1; ids.add(r['config_id']); continue
    keep.append(l)
open(src,'w').writelines(keep); print('purged', n, 'triple-barrier rows (pt/sl floor era)')
for i in ids:
    for f in glob.glob(f'research/clfstudy/.cache/ckpt/{i}_*.safetensors'): os.remove(f)
PY
rm -rf research/clfstudy/.cache/frames/*tb_h*   # label caches were computed with the floored barriers
echo "tb: smoke ($(date))"
python -m research.clfstudy.smoke --arch LSTM_FAM --stride 16 --epochs 3 --pairs LINK/USDT > $LOG/smoke_tb.log 2>&1 || { echo FAIL > $LOG/SMOKE_SCREEN_FAIL; exit 1; }
echo "tb: phase 19 at true +/-2% ($(date))"
python -m research.clfstudy.phase --phase 19 >> $LOG/phase_19b.log 2>&1
python -m research.clfstudy.phase --phase 19 --retry-failed >> $LOG/phase_19b.log 2>&1
python -m research.clfstudy.leaderboard --phase 19 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 19 --ridge-phase 19 >> $LOG/gate_19b.log 2>&1
echo "tb: phase 19 done ($(date))" | tee $LOG/P19B_DONE
for P in 20 21; do
  echo "tb: phase $P ($(date))"
  python -m research.clfstudy.phase --phase $P >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.phase --phase $P --retry-failed >> $LOG/phase_$P.log 2>&1
  python -m research.clfstudy.leaderboard --phase $P >> $LOG/leaderboard.log 2>&1
  python -m research.clfstudy.screen --phase $P --baseline-phase 19 > $LOG/screen_$P.log 2>&1
done
python $LOG/make_phase22.py >> $LOG/screen_22.log 2>&1
python -m research.clfstudy.phase --phase 22 >> $LOG/phase_22.log 2>&1
python -m research.clfstudy.phase --phase 22 --retry-failed >> $LOG/phase_22.log 2>&1
python -m research.clfstudy.leaderboard --phase 22 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.screen --phase 22 --baseline-phase 19 >> $LOG/screen_22.log 2>&1
echo "tb: DONE ($(date))" | tee $LOG/SCREEN_DONE
