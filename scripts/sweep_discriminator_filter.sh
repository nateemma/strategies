#!/bin/zsh
# Sweep gan_synth_discrim_reject_pct on NNNC_DDPM_MLX @ ratio=0.4
# (best no-filter result from the ratio sweep, same baseline as the
# GMM density-filter sweep so the comparison is direct).

set -uo pipefail

ROOT="/Users/philprice95/projects/freqtrade"
cd "$ROOT"

LOG_DIR="${ROOT}/user_data/strategies/scripts/sweep_logs"
mkdir -p "$LOG_DIR"

DDPM_FILE="${ROOT}/user_data/strategies/NNNC/NNNC_DDPM_MLX.py"

DDPM_ORIG_RATIO="0.3"
DDPM_ORIG_DENSITY="0.0"
DDPM_SWEEP_RATIO="0.4"

# The discriminator knob doesn't exist as a line in the strategy file
# yet — we insert it once, sweep it, then remove on exit.

INSERTED_DISCRIM_LINE=0

cleanup() {
  echo "[$(date +%H:%M:%S)] restoring DDPM originals..."
  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_ORIG_RATIO}/" "$DDPM_FILE"
  sed -i '' "s/^    gan_synth_density_reject_pct = .*/    gan_synth_density_reject_pct = ${DDPM_ORIG_DENSITY}/" "$DDPM_FILE"
  if [[ $INSERTED_DISCRIM_LINE -eq 1 ]]; then
    sed -i '' '/^    gan_synth_discrim_reject_pct = .*/d' "$DDPM_FILE"
  fi
}
trap cleanup EXIT INT TERM

# Insert the discriminator knob just after the density knob (idempotent).
if ! grep -q "^    gan_synth_discrim_reject_pct = " "$DDPM_FILE"; then
  sed -i '' "/^    gan_synth_density_reject_pct = /a\\
    gan_synth_discrim_reject_pct = 0.0
" "$DDPM_FILE"
  INSERTED_DISCRIM_LINE=1
fi

run_one() {
  local discrim="$1"

  echo ""
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] NNNC_DDPM_MLX @ ratio=${DDPM_SWEEP_RATIO} discrim_reject_pct=${discrim}"
  echo "==================================================================="

  sed -i '' "s/^    gan_target_ratio = .*/    gan_target_ratio = ${DDPM_SWEEP_RATIO}/" "$DDPM_FILE"
  sed -i '' "s/^    gan_synth_density_reject_pct = .*/    gan_synth_density_reject_pct = 0.0/" "$DDPM_FILE"
  sed -i '' "s/^    gan_synth_discrim_reject_pct = .*/    gan_synth_discrim_reject_pct = ${discrim}/" "$DDPM_FILE"
  grep -E "    gan_target_ratio|    gan_synth_(density|discrim)_reject_pct" "$DDPM_FILE"

  rm -rf "${ROOT}/user_data/strategies/saved_data/NNNC_DDPM_MLX"

  local log="${LOG_DIR}/NNNC_DDPM_MLX_r${DDPM_SWEEP_RATIO}_disc${discrim}.log"
  echo "[$(date +%H:%M:%S)] backtest log → ${log}"

  zsh "${ROOT}/user_data/strategies/scripts/test_strat.sh" -n 720 NNNC NNNC_DDPM_MLX \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] exit code: ${rc}"
}

echo "[$(date +%H:%M:%S)] discriminator-filter sweep started"

for discrim in 0.2 0.3 0.5; do
  run_one "$discrim"
done

echo ""
echo "[$(date +%H:%M:%S)] sweep complete"
