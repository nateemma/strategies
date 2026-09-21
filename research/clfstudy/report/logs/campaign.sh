#!/bin/bash
# ----------------------------------------------------------------------------
# Unattended architecture-search campaign for the forward_mae label.
#
# Runs research.clfstudy.campaign stage by stage. Designed to survive weeks with
# nobody watching:
#   * one trainer at a time (never parallel -- parallel trainers caused an OS
#     out-of-memory on 2026-09-16)
#   * every run already isolated in its own subprocess by phase.py, so a single
#     crashed or OOM-killed run cannot take the campaign down
#   * a watchdog kills a trainer that has produced no new result row for
#     STALL_MIN minutes, so one hung MLX process cannot block the queue forever
#   * fully resumable: every stage is idempotent, rows key on config_id, and
#     re-running the campaign skips everything already in results.jsonl
# ----------------------------------------------------------------------------
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs
RESULTS=research/clfstudy/results/results.jsonl
STALL_MIN=${STALL_MIN:-90}

mkdir -p $LOG

# --- watchdog: kill a trainer that has gone quiet -----------------------------
watchdog() {
  while true; do
    sleep 300
    pgrep -f "research.clfstudy.run" >/dev/null || continue
    if [ -f "$RESULTS" ]; then
      AGE=$(( ($(date +%s) - $(stat -f %m "$RESULTS")) / 60 ))
      if [ "$AGE" -ge "$STALL_MIN" ]; then
        echo "$(date): no new result row for ${AGE}m -- killing stalled trainer" >> $LOG/campaign_watchdog.log
        pkill -f "research.clfstudy.run"
      fi
    fi
  done
}

# --- wait for any earlier chain to finish ------------------------------------
while pgrep -f "chain_p3[0-9]?.sh|clfstudy.phase" >/dev/null; do sleep 120; done

watchdog &
WD=$!
trap "kill $WD 2>/dev/null" EXIT

echo "campaign: START $(date)" | tee -a $LOG/campaign.log
# caffeinate -i holds an idle-sleep assertion for the campaign's lifetime. AC power is
# currently configured with sleep=0 so this is belt-and-braces, but the campaign runs for
# days and a power-setting change or a switch to battery mid-run would otherwise stop it.
caffeinate -i python -m research.clfstudy.campaign --run-all >> $LOG/campaign.log 2>&1
RC=$?
echo "campaign: END rc=$RC $(date)" | tee -a $LOG/campaign.log
echo "rc=$RC $(date)" > $LOG/CAMPAIGN_DONE
