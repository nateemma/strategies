#!/bin/zsh
# Sweep gan_synth_density_reject_pct on NNNC_DDPM_MLX @ ratio=0.4
# (best no-filter result from the ratio sweep).
# Tests whether GMM density filtering of synth — when measured on backtest
# profit rather than on diagnostics — moves the needle.

set -uo pipefail

ROOT="/Users/philprice95/Documents/freqtrade"
cd "$ROOT"

LOG_DIR="${ROOT}/user_data/strategies/scripts/sweep_logs"
mkdir -p "$LOG_DIR"

DDPM_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_DDPM_MLX.py"

DDPM_ORIG_RATIO="0.3"
DDPM_ORIG_REJECT="0.0"
DDPM_SWEEP_RATIO="0.4"  # best from no-filter sweep

cleanup() {
  echo "[$(date +%H:%M:%S)] restoring DDPM originals..."
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_ORIG_RATIO}/" "$DDPM_FILE"
  sed -i '' "s/^    gan_synth_density_reject_pct = .*/    gan_synth_density_reject_pct = ${DDPM_ORIG_REJECT}/" "$DDPM_FILE"
}
trap cleanup EXIT INT TERM

run_one() {
  local reject="$1"

  echo ""
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] NNNC_DDPM_MLX @ ratio=${DDPM_SWEEP_RATIO} reject_pct=${reject}"
  echo "==================================================================="

  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_SWEEP_RATIO}/" "$DDPM_FILE"
  sed -i '' "s/^    gan_synth_density_reject_pct = .*/    gan_synth_density_reject_pct = ${reject}/" "$DDPM_FILE"
  grep -E "    gan_target_ratio|    gan_synth_density_reject_pct" "$DDPM_FILE"

  rm -rf "${ROOT}/user_data/strategies/saved_data/NNNC_DDPM_MLX"

  local log="${LOG_DIR}/NNNC_DDPM_MLX_r${DDPM_SWEEP_RATIO}_rej${reject}.log"
  echo "[$(date +%H:%M:%S)] backtest log → ${log}"

  zsh "${ROOT}/user_data/strategies/scripts/test_strat.sh" -n 720 NNNC NNNC_DDPM_MLX \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] exit code: ${rc}"
}

echo "[$(date +%H:%M:%S)] density-filter sweep started"

for reject in 0.2 0.3 0.5; do
  run_one "$reject"
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
