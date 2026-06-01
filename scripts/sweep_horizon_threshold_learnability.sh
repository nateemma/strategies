#!/bin/zsh
# Sweep HORIZON × threshold combinations through DebugSignalLearnability.py
# and aggregate the per-(horizon, threshold) MCC results into one table.
#
# DebugSignalLearnability sweeps threshold within a single horizon. This
# wrapper loops over horizons, calls the tool for each, and concatenates
# the per-horizon CSVs into a single table keyed by (horizon, threshold).
#
# Production reference points:
#   - HORIZON = 3
#   - MIN_BUY_GAIN_THRESHOLD = 0.005
#   - MIN_SELL_LOSS_THRESHOLD = 0.005
#   - TRAINING_TYPE = 17 (gbb)
#
# Sweep neighborhood:
#   - horizons: 2 3 4 5 6 8
#   - thresholds: 0.003 0.004 0.005 0.007 0.010 0.013
#   - method: 17 (gbb)
#   - side: both (combined buy + sell)
#
# Pair selection: defaults to XRP/USDT (highest-MCC pair in the recent
# backtest). Pass --pair or set PAIR env var to change.

set -uo pipefail

ROOT="/Users/philprice95/Documents/freqtrade"
cd "$ROOT"

PAIR="${PAIR:-XRP_USDT}"
TIMEFRAME="${TIMEFRAME:-15m}"
METHOD="${METHOD:-17}"
SIDE="${SIDE:-both}"
MAX_BARS="${MAX_BARS:-0}"   # 0 = use full history

HORIZONS=(2 3 4 5 6 8)
THRESHOLDS=(0.003 0.004 0.005 0.007 0.010 0.013)

OUT_DIR="/tmp/learnability_sweep"
mkdir -p "$OUT_DIR"
SUMMARY="${OUT_DIR}/summary.csv"

echo "horizon,threshold,method,side,mcc,signal_freq,expected_gain,expected_total_profit" \
  > "$SUMMARY"

THRESHOLD_ARGS="${THRESHOLDS[*]}"

echo "[$(date +%H:%M:%S)] sweep started — pair=${PAIR} method=${METHOD} side=${SIDE}"
echo "  horizons: ${HORIZONS[*]}"
echo "  thresholds: ${THRESHOLD_ARGS}"
echo ""

for h in "${HORIZONS[@]}"; do
  csv="${OUT_DIR}/h${h}.csv"
  echo "[$(date +%H:%M:%S)] horizon=${h} → ${csv}"

  # Build the command, only adding --max-bars when MAX_BARS > 0.
  # zsh's ${:+} substitution fires even when MAX_BARS=0, which the tool
  # rejects with "unrecognized arguments: --max-bars 0".
  cmd=(python3 user_data/strategies/Debug/DebugSignalLearnability.py
       --pair "${PAIR}"
       --timeframe "${TIMEFRAME}"
       --horizon "${h}"
       --methods "${METHOD}"
       --thresholds ${=THRESHOLD_ARGS}
       --side "${SIDE}"
       --save-csv "${csv}")
  if [[ "${MAX_BARS}" -gt 0 ]]; then
    cmd+=(--max-bars "${MAX_BARS}")
  fi
  "${cmd[@]}" > "${OUT_DIR}/h${h}.log" 2>&1

  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  [WARN] horizon=${h} exited ${rc} — see ${OUT_DIR}/h${h}.log"
    continue
  fi

  # Append rows from the per-horizon CSV to the summary, prefixed by horizon.
  if [[ -f "$csv" ]]; then
    tail -n +2 "$csv" | awk -F',' -v h="$h" '{print h","$0}' >> "$SUMMARY"
  fi
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
echo "Combined summary: ${SUMMARY}"
echo ""
echo "Top combinations by MCC:"
echo "horizon  threshold  mcc       signal_freq  expected_gain  expected_total_profit"
echo "-----------------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k5 -gr | head -10 | \
  awk -F',' '{printf "%-8s %-9s %-9s %-12s %-14s %s\n", $1, $2, $5, $6, $7, $8}'
