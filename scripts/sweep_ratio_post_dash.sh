#!/bin/zsh
# Sweep gan_target_ratio on NNNC_DDPM_MLX after the DASH-removed retrain.
# Tests whether the ratio=0.3 → 0.5 improvement (+22% Calmar) keeps
# climbing with cleaner training data. Run this AFTER:
#   1. DASH removed from pair list
#   2. CreateScalers regenerated
#   3. CreateMTDDPM retrained (cleaner training data)
#   4. CreateAutoencoderFilter retrained (cleaner manifold)
#
# Sweep points:
#   0.5 — confirm the pre-DASH-removal win replicates
#   0.7 — push past the previous best
#   0.8 — aggressive, historic NNMT_MT_DDPM production ratio
#
# Each run edits the ratio in the strategy file, drops the classifier
# saved_data so it retrains under the new ratio, captures the backtest
# log, and restores the original ratio on exit (trap covers errors too).

set -uo pipefail

ROOT="/Users/philprice95/Documents/freqtrade"
cd "$ROOT"

LOG_DIR="${ROOT}/user_data/strategies/scripts/sweep_logs"
mkdir -p "$LOG_DIR"

DDPM_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_DDPM_MLX.py"
DDPM_ORIG_RATIO="0.5"  # set this to whatever you're at when you start the sweep

cleanup() {
  echo "[$(date +%H:%M:%S)] restoring DDPM original ratio (${DDPM_ORIG_RATIO})..."
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_ORIG_RATIO}/" "$DDPM_FILE"
}
trap cleanup EXIT INT TERM

run_one() {
  local ratio="$1"

  echo ""
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] NNNC_DDPM_MLX @ gan_target_ratio=${ratio}"
  echo "==================================================================="

  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${ratio}/" "$DDPM_FILE"
  grep -E "    gan_target_ratio|    gan_synth_autoencoder_threshold" "$DDPM_FILE"

  rm -rf "${ROOT}/user_data/strategies/saved_data/NNNC_DDPM_MLX"

  local log="${LOG_DIR}/NNNC_DDPM_MLX_postdash_r${ratio}.log"
  echo "[$(date +%H:%M:%S)] backtest log → ${log}"

  zsh "${ROOT}/user_data/strategies/scripts/test_strat.sh" -n 720 NNNC NNNC_DDPM_MLX \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] exit code: ${rc}"

  # Pull out the headline metrics so the sweep summary is greppable.
  local profit=$(grep -E "Total profit %" "$log" | head -1 | awk -F'│' '{print $3}' | tr -d ' ')
  local dd=$(grep -E "Max % of account underwater\b" "$log" | head -1 | awk -F'│' '{print $3}' | tr -d ' ')
  local calmar=$(grep -E "Calmar \(closed" "$log" | head -1 | awk -F'│' '{print $3}' | tr -d ' ')
  local sharpe=$(grep -E "Sharpe \(closed" "$log" | head -1 | awk -F'│' '{print $3}' | tr -d ' ')
  echo "[$(date +%H:%M:%S)] result: profit=${profit} dd=${dd} calmar=${calmar} sharpe=${sharpe}"
}

echo "[$(date +%H:%M:%S)] post-DASH ratio sweep started"

for ratio in 0.5 0.7 0.8; do
  run_one "$ratio"
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
