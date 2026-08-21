#!/bin/zsh
# Targeted mini-sweep based on the v1 Pareto frontier finding (2026-05-30).
#
# v1 found:
#   - Non-HIGH-aug_risk combos with MCC > 0.55 are all at H=24
#   - LINK H=24 thr=0.010 was the closest to viable (MCC 0.61, EV 2.98%, MEDIUM)
#   - Untested: thr between 0.005 and 0.010 at H=24; longer horizons at thr 0.010-0.013
#
# v2 fills that gap with:
#   H = [24, 36, 48]      — extend horizon for higher EV without killing density
#   thr = [0.007, 0.010, 0.013]   — fill the 0.005-0.010 gap, plus one higher
#
# Same pairs and viability criteria as v1.

set -uo pipefail

ROOT="/Users/philprice95/projects/freqtrade"
cd "$ROOT"

TIMEFRAME="${TIMEFRAME:-15m}"
METHOD="${METHOD:-17}"
SIDE="${SIDE:-both}"
MAX_BARS="${MAX_BARS:-0}"

PAIRS=("${(@s/,/)${PAIRS:-XRP_USDT,SOL_USDT,LINK_USDT}}")
HORIZONS=(24 36 48)
THRESHOLDS=(0.007 0.010 0.013)

OUT_ROOT="/tmp/learnability_viable_v2"
mkdir -p "$OUT_ROOT"

SUMMARY="${OUT_ROOT}/summary.csv"
echo "pair,horizon,method,method_id,side,threshold,bb_width_threshold,n_signals,n_total,mcc,ev_per_signal_pct,score,aug_risk,error" \
  > "$SUMMARY"

THRESHOLD_ARGS="${THRESHOLDS[*]}"

echo "[$(date +%H:%M:%S)] viable-combo sweep v2 started"
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

echo "Full results:"
echo "pair         H    thr     side  N_sig   MCC      EV%    aug_risk"
echo "----------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k1,1 -k2,2n -k5,5 -k6,6g | \
  awk -F',' '{printf "%-12s %-4s %-7s %-5s %-6s %-8s %-7s %s\n", \
    $1, $2, $6, $5, $8, $10, $11, $13}'

echo ""
echo "=== VIABLE COMBOS (MCC>0.30 AND aug_risk!=HIGH AND EV>3.0) ==="
viable=$(tail -n +2 "$SUMMARY" | awk -F',' '
  $10 != "" && $10 + 0 > 0.30 && $11 + 0 > 3.0 && $13 != "HIGH" {
    printf "%-12s H=%-3s thr=%-6s %-5s  MCC=%.2f  EV=%.2f%%  %s\n", $1, $2, $6, $5, $10, $11, $13
  }')
if [[ -z "$viable" ]]; then
  echo "  (none — gbb labeler genuinely exhausted)"
else
  echo "$viable"
fi

echo ""
echo "=== Pareto frontier — non-HIGH aug_risk, ranked by EV ==="
tail -n +2 "$SUMMARY" | awk -F',' '
  $13 != "HIGH" && $11 != "" {
    printf "%-12s H=%-3s thr=%-6s %-5s  MCC=%.2f  EV=%.2f%%  %s\n", $1, $2, $6, $5, $10, $11, $13
  }' | sort -k6 -t= -gr | head -15
