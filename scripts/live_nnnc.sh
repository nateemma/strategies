#!/bin/zsh
# live_nnnc.sh — the 036 live NNNC dry-run bot (specs/036-realistic-nnnc-pnl, research R2).
#
#   zsh user_data/strategies/scripts/live_nnnc.sh start|stop|status|--check   # from the repo root
#
# Isolated from the momentum dry-run (port 8080, tradesv3.dryrun.sqlite): its own config
# (config_live_nnnc.json, port 8081), database, log and pid file. FT_MODEL_EVAL_STRICT=1 makes a
# missing model fatal instead of training live. `stop` signals ONLY the pid in its own pid file,
# after checking that the process command line names config_live_nnnc.json. It never pkills.
set -u
cd "$(dirname "$0")/../../.."   # -> repo root
[ -f "$HOME/.freqtrade_env" ] && { set -a; . "$HOME/.freqtrade_env"; set +a; }

CONFIG=user_data/strategies/config/config_live_nnnc.json
DB=${LIVE_NNNC_DB:-tradesv3.live_nnnc.sqlite}
PIDFILE=${LIVE_NNNC_PIDFILE:-user_data/logs/live_nnnc.pid}
LOG=user_data/logs/live_nnnc.log
export LIVE_NNNC_DB=$DB LIVE_NNNC_PIDFILE=$PIDFILE

check() {
  PYTHONPATH=user_data/strategies .venv/bin/python -m research.live_pnl.launch_check
}

own_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid=$(cat "$PIDFILE")
  ps -p "$pid" -o command= 2>/dev/null | grep -q "config_live_nnnc.json" || return 1
  echo "$pid"
}

case "${1:-}" in
  --check)
    check
    ;;
  start)
    out=$(check) || exit 2
    eval "$out"   # BOT_CLASS=..., RUN_ID=...
    mkdir -p user_data/logs
    FT_MODEL_EVAL_STRICT=1 nohup .venv/bin/freqtrade trade --dry-run \
      -c "$CONFIG" --db-url "sqlite:///$DB" \
      --strategy-path user_data/strategies/NNNC -s "$BOT_CLASS" \
      --logfile "$LOG" > user_data/logs/live_nnnc.stdout 2>&1 &
    echo $! > "$PIDFILE"
    echo "started $BOT_CLASS (run $RUN_ID) pid $(cat "$PIDFILE") port 8081 db $DB log $LOG"
    ;;
  stop)
    pid=$(own_pid) || { echo "not running"; exit 0; }
    kill "$pid" && echo "stopped pid $pid"
    ;;
  status)
    if pid=$(own_pid); then
      echo "running pid $pid"
      curl -s -o /dev/null -w "api 8081: HTTP %{http_code}\n" http://127.0.0.1:8081/api/v1/ping || true
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: live_nnnc.sh start|stop|status|--check" >&2; exit 2
    ;;
esac
