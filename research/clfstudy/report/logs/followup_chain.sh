#!/bin/bash
# Single sequential follow-up chain installed by the main session (2026-09-16).
# Waits for the phase-3 trainer/restart loop to finish, then runs the remaining steps ONE AT A TIME.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate
export PYTHONPATH=.
LOG=research/clfstudy/report/logs
echo "chain: waiting for phase 3 ($(date))"
while pgrep -f "research.clfstudy.phase --phase 3" >/dev/null || ps -p 74474 >/dev/null 2>&1; do sleep 300; done
echo "chain: phase 3 trainer gone ($(date)); retrying failed once"
python -m research.clfstudy.phase --phase 3 --retry-failed >> $LOG/phase_3_retry.log 2>&1
echo "chain: running phase 2 for the fixed ROCKET config ($(date))"
python -m research.clfstudy.phase --phase 2 >> $LOG/phase_2_rocket_fix.log 2>&1
echo "chain: leaderboards ($(date))"
python -m research.clfstudy.leaderboard --phase 2 >> $LOG/leaderboard.log 2>&1
python -m research.clfstudy.leaderboard --phase 3 >> $LOG/leaderboard.log 2>&1
echo "chain: DONE ($(date))" | tee research/clfstudy/report/logs/CHAIN_DONE
