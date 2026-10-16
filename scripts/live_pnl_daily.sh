#!/bin/zsh
# live_pnl_daily.sh — 036 daily duties for the live NNNC dry-run (research/live_pnl, task T017).
#
#   zsh user_data/strategies/scripts/live_pnl_daily.sh [YYYY-MM-DD]   # default: yesterday (UTC)
#
# Runs, in order:
#   1. parity for the day: download candles, run a cumulative backtest, and compare with the live
#      trades;
#   2. drift: the live BUY rate against target;
#   3. gate: days recorded and orders per pair.
#
# Writes only research/live_pnl/{parity/<day>.json, drift.json, gate.json}. Commits nothing; the
# weekly summary is committed by hand. Parity refuses to download while a training or backtest
# chain is running; that is logged and the day can be re-run later with its date as the argument.
#
# Cron (user crontab; local time, after the UTC day has closed):
#   40 17 * * * zsh /Users/philprice95/projects/freqtrade/user_data/strategies/scripts/live_pnl_daily.sh \
#     >> /Users/philprice95/projects/freqtrade/user_data/logs/live_pnl_daily.log 2>&1
set -u
cd "$(dirname "$0")/.."            # -> user_data/strategies
[ -f "$HOME/.freqtrade_env" ] && { set -a; . "$HOME/.freqtrade_env"; set +a; }
PY=../../.venv/bin/python
export PYTHONPATH=.
DAY=${1:-$(date -u -v-1d +%Y-%m-%d)}

echo "=== $(date -u +%FT%TZ) live_pnl_daily day=$DAY"
if ! ps -p "$(cat ../logs/live_nnnc.pid 2>/dev/null || echo 0)" -o command= 2>/dev/null | grep -q config_live_nnnc.json; then
  echo "WARNING: live NNNC bot is not running (scripts/live_nnnc.sh status)"
fi
$PY -m research.live_pnl.parity --day "$DAY" || echo "parity FAILED for $DAY (exit $?)"
$PY -m research.live_pnl.drift > /dev/null || echo "drift FAILED"
$PY -m research.live_pnl.gate || echo "gate FAILED"
echo "=== done $(date -u +%FT%TZ)"
