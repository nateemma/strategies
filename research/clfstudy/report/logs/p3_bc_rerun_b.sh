#!/bin/bash
# Continuation of the phase-3 B/C re-run without Mamba (48 GB Metal peak, 68 min/run; excluded on window C).
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "p3bc-b: resume (Sat Sep 19 05:10:42 PDT 2026)"
python -m research.clfstudy.phase --phase 3 --only d2bbc0d7fb15,098ca2f95a2e,f37089f602fe,076666f04306,60f40876941b,ea8d90054291,0b31c7bb9404,cf6b6ccb530a,52292292ef8b,666675f56e77,f1f70fadabbb,3cd40f90604c,0875fd5bd454,3a4ddb6c4092,59d72ed6d750,e645063be7bc,3d09879bf74b,d7fcd89ef3d6,48b806ceba9b,470786990c9e,8d1446f2a484,225a954527c8,558e2f29084d >> $LOG/phase_3_bc_rerun.log 2>&1
python -m research.clfstudy.phase --phase 3 --only d2bbc0d7fb15,098ca2f95a2e,f37089f602fe,076666f04306,60f40876941b,ea8d90054291,0b31c7bb9404,cf6b6ccb530a,52292292ef8b,666675f56e77,f1f70fadabbb,3cd40f90604c,0875fd5bd454,3a4ddb6c4092,59d72ed6d750,e645063be7bc,3d09879bf74b,d7fcd89ef3d6,48b806ceba9b,470786990c9e,8d1446f2a484,225a954527c8,558e2f29084d --retry-failed >> $LOG/phase_3_bc_rerun.log 2>&1
python -m research.clfstudy.leaderboard --phase 3 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.gate --phase 3 >> $LOG/gate_3_after_bc_rerun.log 2>&1
echo "p3bc: DONE ($(date))" | tee $LOG/P3_BC_RERUN_DONE
