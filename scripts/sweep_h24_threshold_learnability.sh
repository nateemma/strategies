#!/bin/zsh
# Sweep threshold combinations at HORIZON=24 through DebugSignalLearnability.
#
# Rationale: HORIZON=3 sweeps (sweep_horizon_threshold_learnability.sh) used
# thresholds 0.003-0.013 because at 3 bars even small moves matter. At
# HORIZON=24 (6h forward), the window is 8× longer and meaningful labels
# need correspondingly larger gain targets. Sweeps tests 0.005 → 0.05 to
# find the threshold where the TP:SL ratio (stop_loss ≈ -3%) starts to
# favour the trade.
#
# Runs across XRP/SOL/LINK like the prior sweep so results are comparable.

set -uo pipefail

ROOT="/Users/philprice95/Documents/freqtrade"
cd "$ROOT"

TIMEFRAME="${TIMEFRAME:-15m}"
METHOD="${METHOD:-17}"
SIDE="${SIDE:-both}"
MAX_BARS="${MAX_BARS:-0}"

PAIRS=("${(@s/,/)${PAIRS:-XRP_USDT,SOL_USDT,LINK_USDT}}")
HORIZON=24
THRESHOLDS=(0.005 0.010 0.015 0.020 0.030 0.050)

OUT_ROOT="/tmp/learnability_sweep_h24"
mkdir -p "$OUT_ROOT"

SUMMARY="${OUT_ROOT}/summary.csv"
echo "pair,horizon,threshold,method,side,mcc,signal_freq,expected_gain,expected_total_profit" \
  > "$SUMMARY"

THRESHOLD_ARGS="${THRESHOLDS[*]}"

echo "[$(date +%H:%M:%S)] H=24 sweep started"
echo "  pairs: ${PAIRS[*]}"
echo "  horizon: ${HORIZON}"
echo "  thresholds: ${THRESHOLD_ARGS}"
echo "  method: ${METHOD}  side: ${SIDE}"
echo ""

for pair in "${PAIRS[@]}"; do
  pair_dir="${OUT_ROOT}/${pair//\//_}"
  mkdir -p "$pair_dir"
  csv="${pair_dir}/h${HORIZON}.csv"
  log="${pair_dir}/h${HORIZON}.log"

  echo "[$(date +%H:%M:%S)] pair=${pair} → ${csv}"

  cmd=(python3 user_data/strategies/Debug/DebugSignalLearnability.py
       --pair "${pair}"
       --timeframe "${TIMEFRAME}"
       --horizon "${HORIZON}"
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
    echo "  [WARN] pair=${pair} exited ${rc} — see ${log}"
    continue
  fi

  if [[ -f "$csv" ]]; then
    tail -n +2 "$csv" | awk -F',' -v p="$pair" -v h="$HORIZON" \
      '{print p","h","$0}' >> "$SUMMARY"
  fi
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
echo "Combined summary: ${SUMMARY}"
echo ""
echo "Per-(pair, threshold) results:"
echo "pair         thr      mcc       signal_freq  exp_gain    exp_total"
echo "----------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k1,1 -k3,3g | \
  awk -F',' '{printf "%-12s %-8s %-9s %-12s %-11s %s\n", $1, $3, $6, $7, $8, $9}'

echo ""
echo "Top 10 combinations by MCC:"
echo "pair         thr      mcc       signal_freq  exp_gain    exp_total"
echo "----------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k6 -gr | head -10 | \
  awk -F',' '{printf "%-12s %-8s %-9s %-12s %-11s %s\n", $1, $3, $6, $7, $8, $9}'

echo ""
echo "Top 10 combinations by expected_total_profit:"
echo "pair         thr      mcc       signal_freq  exp_gain    exp_total"
echo "----------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | sort -t',' -k9 -gr | head -10 | \
  awk -F',' '{printf "%-12s %-8s %-9s %-12s %-11s %s\n", $1, $3, $6, $7, $8, $9}'
