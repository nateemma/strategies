#!/bin/zsh
# Sweep gan_target_ratio for NNNC_DDPM_MLX, NNNC_CGP_MLX, NNNC_WGAN_MLX.
# Each run: edit ratio in strategy file, drop classifier saved data, run 720d
# backtest, capture log. Restore original ratios on exit (incl. errors).

set -uo pipefail

ROOT="/Users/philprice95/projects/freqtrade"
cd "$ROOT"

LOG_DIR="${ROOT}/user_data/strategies/scripts/sweep_logs"
mkdir -p "$LOG_DIR"

DDPM_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_DDPM_MLX.py"
CGP_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_CGP_MLX.py"
WGAN_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_WGAN.py"

DDPM_ORIG="0.3"
CGP_ORIG="0.4"
WGAN_ORIG="0.3"

cleanup() {
  echo "[$(date +%H:%M:%S)] restoring original ratios..."
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_ORIG}/" "$DDPM_FILE"
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${CGP_ORIG}/" "$CGP_FILE"
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${WGAN_ORIG}/" "$WGAN_FILE"
}
trap cleanup EXIT INT TERM

run_one() {
  local variant="$1"
  local file="$2"
  local ratio="$3"

  echo ""
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] ${variant} @ ratio=${ratio}"
  echo "==================================================================="

  # Edit ratio
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${ratio}/" "$file"
  grep "    gan_target_ratio" "$file"

  # Drop classifier so it retrains under new ratio. Keep GAN model.
  rm -rf "${ROOT}/user_data/strategies/saved_data/${variant}"

  # Run backtest
  local log="${LOG_DIR}/${variant}_r${ratio}.log"
  echo "[$(date +%H:%M:%S)] backtest log → ${log}"

  zsh "${ROOT}/user_data/strategies/scripts/test_strat.sh" -n 720 NNNC "${variant}" \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] exit code: ${rc}"
  return $rc
}

# Variants × ratios. Skip ratios that equal each variant's current default
# (those are baselines you've effectively already run at H=6; if you want
# an H=3 re-baseline, re-include them).
echo "[$(date +%H:%M:%S)] sweep started"

for ratio in 0.1 0.2 0.4; do
  run_one "NNNC_DDPM_MLX" "$DDPM_FILE" "$ratio"
done

for ratio in 0.1 0.2 0.3; do
  run_one "NNNC_CGP_MLX" "$CGP_FILE" "$ratio"
done

for ratio in 0.1 0.2 0.4; do
  run_one "NNNC_WGAN_MLX" "$WGAN_FILE" "$ratio"
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
