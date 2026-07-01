#!/usr/bin/env zsh
#
# walk_forward.sh — rolling walk-forward for a Basket strategy.
#
# For each fold: hyperopt on the IS window (WalletCalmarHyperOptLoss), then
# backtest the tuned params on IS and on the following (untouched) OOS window,
# and print an IS-vs-OOS table of profit / wallet-balance Calmar / drawdown.
# The <Strategy>.json written by each hyperopt is the hand-off to its backtests;
# any pre-existing json is backed up and restored at the end.
#
# Usage: zsh walk_forward.sh [-e epochs] [-f folds] [-c config]
#                            [-s startYYYYMMDD] [-n is_days] [-m oos_days] <Strategy>

set -e
cd "$(dirname "$0")/../../.."   # -> freqtrade repo root

config="user_data/strategies/config/config_basket.json"
spath="user_data/strategies/Basket"
loss="WalletCalmarHyperOptLoss"
epochs=200
folds=4
start="20240610"
is_days=365
oos_days=90

while getopts ":e:f:c:s:n:m:" opt; do
  case $opt in
    e) epochs="$OPTARG" ;;
    f) folds="$OPTARG" ;;
    c) config="$OPTARG" ;;
    s) start="$OPTARG" ;;
    n) is_days="$OPTARG" ;;
    m) oos_days="$OPTARG" ;;
    *) echo "bad option"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
strategy="$1"
[ -z "$strategy" ] && { echo "Usage: zsh walk_forward.sh [opts] <Strategy>"; exit 2; }

data_end="20260531"

add_days() {  # $1=YYYYMMDD  $2=n
  if [ "$(uname)" = "Darwin" ]; then
    date -j -v+${2}d -f "%Y%m%d" "$1" "+%Y%m%d"
  else
    date -d "$1 + $2 days" "+%Y%m%d"
  fi
}

# metric grep helpers (read a backtest log)
prof() { grep -iE "Total profit %" "$1" 2>/dev/null | grep -oE "\-?[0-9.]+%" | head -1; }
cal()  { grep -iE "Calmar \(daily wallet" "$1" 2>/dev/null | grep -oE "\-?[0-9.]+" | head -1; }
ddn()  { grep -iE "Max % of account underwater .balance" "$1" 2>/dev/null | grep -oE "[0-9.]+%" | head -1; }

# make the loss available; back up any existing tuned json
cp user_data/strategies/hyperopts/${loss}.py user_data/hyperopts/ 2>/dev/null || true
json="${spath}/${strategy}.json"
[ -f "$json" ] && cp "$json" "/tmp/wf_backup_${strategy}.json"

echo "Walk-forward: $strategy  epochs=$epochs folds=$folds  IS=${is_days}d OOS=${oos_days}d"
printf "%-4s  %-19s %-8s %-8s   %-19s %-9s %-8s %-8s\n" \
       fold "IS range" IS_prof IS_cal "OOS range" OOS_prof OOS_cal OOS_dd

for k in $(seq 0 $((folds - 1))); do
  is_start=$(add_days "$start" $((k * oos_days)))
  is_end=$(add_days "$is_start" "$is_days")
  oos_start="$is_end"
  oos_end=$(add_days "$oos_start" "$oos_days")
  [ "$oos_end" -gt "$data_end" ] && oos_end="$data_end"
  [ "$oos_start" -ge "$data_end" ] && break

  L="/tmp/wf_${strategy}_${k}"
  freqtrade hyperopt -c "$config" --strategy-path "$spath" --strategy "$strategy" \
    --hyperopt-loss "$loss" --spaces buy --epochs "$epochs" --min-trades 1 \
    --timerange="${is_start}-${is_end}" > "${L}_hopt.log" 2>&1 || true
  freqtrade backtesting -c "$config" --strategy-path "$spath" --strategy "$strategy" \
    --timerange="${is_start}-${is_end}" > "${L}_is.log" 2>&1 || true
  freqtrade backtesting -c "$config" --strategy-path "$spath" --strategy "$strategy" \
    --timerange="${oos_start}-${oos_end}" > "${L}_oos.log" 2>&1 || true

  printf "%-4s  %-19s %-8s %-8s   %-19s %-9s %-8s %-8s\n" \
    "$k" "${is_start}-${is_end}" "$(prof ${L}_is.log)" "$(cal ${L}_is.log)" \
    "${oos_start}-${oos_end}" "$(prof ${L}_oos.log)" "$(cal ${L}_oos.log)" "$(ddn ${L}_oos.log)"
done

# restore the user's original tuned json (if any)
if [ -f "/tmp/wf_backup_${strategy}.json" ]; then
  cp "/tmp/wf_backup_${strategy}.json" "$json"
  echo "(restored original ${strategy}.json)"
fi
echo "DONE"
