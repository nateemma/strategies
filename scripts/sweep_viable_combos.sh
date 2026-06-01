#!/bin/zsh
# Sweep (H, threshold, pair) combinations through DebugSignalLearnability
# with the new aug_risk + MLP-MCC output, then filter for combos that satisfy
# all three viability criteria simultaneously:
#   - MLP MCC > 0.30
#   - aug_risk != HIGH  (LOW or MEDIUM only)
#   - EV per signal > 3.0%
#
# If any combo passes all three across all three pairs, that's a real
# candidate for a backtest. If none do, the gbb labeler is structurally
# exhausted in this neighborhood.

set -uo pipefail

ROOT="/Users/philprice95/Documents/freqtrade"
cd "$ROOT"

TIMEFRAME="${TIMEFRAME:-15m}"
METHOD="${METHOD:-17}"
SIDE="${SIDE:-both}"
MAX_BARS="${MAX_BARS:-0}"

PAIRS=("${(@s/,/)${PAIRS:-XRP_USDT,SOL_USDT,LINK_USDT}}")
HORIZONS=(3 6 12 24)
THRESHOLDS=(0.005 0.010 0.015 0.020)

OUT_ROOT="/tmp/learnability_viable"
mkdir -p "$OUT_ROOT"

SUMMARY="${OUT_ROOT}/summary.csv"
echo "pair,horizon,method,method_id,side,threshold,bb_width_threshold,n_signals,n_total,mcc,ev_per_signal_pct,score,aug_risk,error" \
  > "$SUMMARY"

THRESHOLD_ARGS="${THRESHOLDS[*]}"

echo "[$(date +%H:%M:%S)] viable-combo sweep started"
echo "  pairs:      ${PAIRS[*]}"
echo "  horizons:   ${HORIZONS[*]}"
echo "  thresholds: ${THRESHOLD_ARGS}"
echo "  method: ${METHOD}  side: ${SIDE}"
echo ""

for pair in "${PAIRS[@]}"; do
  pair_dir="${OUT_ROOT}/${pair//\//_}"
  mkdir -p "$pair_dir"
  for h in "${HORIZONS[@]}"; do
    csv="${pair_dir}/h${h}.csv"
    log="${pair_dir}/h${h}.log"

    echo "[$(date +%H:%M:%S)] pair=${pair} horizon=${h}"

    cmd=(python3 user_data/strategies/Debug/DebugSignalLearnability.py
         --pair "${pair}"
         --timeframe "${TIMEFRAME}"
         --horizon "${h}"
         --methods "${METHOD}"
         --thresholds ${=THRESHOLD_ARGS}
         --side "${SIDE}"
         --save-csv "${csv}")
    if [[ "${MAX_BARS}" -gt 0 ]]; then
      cmd+=(--max-bars "${MAX_BARS}")
    fi
    "${cmd[@]}" > "${log}" 2>&1

    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  [WARN] pair=${pair} horizon=${h} exited ${rc} — see ${log}"
      continue
    fi

    if [[ -f "$csv" ]]; then
      tail -n +2 "$csv" | awk -F',' -v p="$pair" -v h="$h" \
        '{print p","h","$0}' >> "$SUMMARY"
    fi
  done
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
echo "Combined summary: ${SUMMARY}"
echo ""

# Print the full table sorted by (pair, horizon, threshold)
echo "Full results:"
echo "pair         H    thr     side  N_sig   MCC      EV%    aug_risk"
echo "----------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k1,1 -k2,2n -k5,5 -k6,6g | \
  awk -F',' '{printf "%-12s %-4s %-7s %-5s %-6s %-8s %-7s %s\n", \
    $1, $2, $6, $5, $8, $10, $11, $13}'

echo ""
echo "=== VIABLE COMBOS (MCC>0.30 AND aug_risk!=HIGH AND EV>3.0) ==="
echo "pair         H    thr     side  N_sig   MCC      EV%    aug_risk"
echo "----------------------------------------------------------------------"
viable=$(tail -n +2 "$SUMMARY" | awk -F',' '
  $10 != "" && $10 + 0 > 0.30 && $11 + 0 > 3.0 && $13 != "HIGH" {
    printf "%-12s %-4s %-7s %-5s %-6s %-8s %-7s %s\n", $1, $2, $6, $5, $8, $10, $11, $13
  }')

if [[ -z "$viable" ]]; then
  echo "  (none — gbb labeler exhausted in this neighborhood)"
else
  echo "$viable"
fi

echo ""
echo "=== Per-(H, threshold) — viable in all 3 pairs? ==="
echo "Counts per (horizon, threshold, side) of how many pairs are viable:"
tail -n +2 "$SUMMARY" | awk -F',' '
  $10 != "" && $10 + 0 > 0.30 && $11 + 0 > 3.0 && $13 != "HIGH" {
    key = $2 "_" $6 "_" $5
    count[key]++
  }
  END {
    for (k in count) printf "  %s: %d pairs\n", k, count[k]
  }' | sort
