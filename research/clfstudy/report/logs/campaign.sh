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

# --- single-instance guard ---------------------------------------------------
# Added 2026-09-21 after a restart left TWO drivers running: `pkill -f campaign.sh`
# kills this script but NOT the `python -m research.clfstudy.campaign` it spawned, so
# relaunching produced two drivers and two concurrent trainers -- exactly the condition
# that caused the 2026-09-16 out-of-memory. Refuse to start instead, and say how to stop
# the old one. $$ and the watchdog subshell are excluded so we never match ourselves.
OTHERS=$(pgrep -f "research.clfstudy.campaign --run-all" | grep -v "^$$\$" || true)
if [ -n "$OTHERS" ]; then
  echo "campaign driver already running (pids: $(echo $OTHERS | tr '\n' ' '))." | tee -a $LOG/campaign.log
  echo "refusing to start a second one. To restart cleanly:" | tee -a $LOG/campaign.log
  echo "  pkill -f 'research.clfstudy.campaign --run-all'; pkill -f 'research.clfstudy.run'" | tee -a $LOG/campaign.log
  echo "  then wait 5s and relaunch this script." | tee -a $LOG/campaign.log
  exit 3
fi

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

# --- wait for campaign.py, then prove it loads before committing days to it ---
# This script is launched before campaign.py is finished writing, deliberately: it means
# the campaign starts on its own and does not depend on anyone being around to press go.
for i in $(seq 1 720); do [ -f research/clfstudy/campaign.py ] && break; sleep 60; done
if [ ! -f research/clfstudy/campaign.py ]; then
  echo "campaign.py never appeared" | tee $LOG/CAMPAIGN_FAIL; exit 1
fi
# A dry run of every stage is cheap and catches an import error, a bad arch name or a
# malformed hp block BEFORE the campaign spends days acting on it.
if ! python -m research.clfstudy.campaign --dry-run --stage A >> $LOG/campaign.log 2>&1; then
  echo "campaign --dry-run failed; refusing to start" | tee $LOG/CAMPAIGN_FAIL
  exit 1
fi

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
