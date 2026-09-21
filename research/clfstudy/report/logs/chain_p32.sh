#!/bin/bash
# Phase 32: production LSTM vs its NOCONV/NORES ablations vs plain LSTM_FAM,
# seq_len 64, forward_mae, WINDOW A ONLY. Curiosity probe -- gates nothing.
#
# NOTE: --max-runs alone will NOT give window A across several configs.
# phase.expand_runs is config-major (for cfg: for window: for seed), so a global
# cap would spend itself on the first config's three windows. Instead loop the
# config_ids and cap each at 2, which is that config's window A seeds 0,1.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
LOG=research/clfstudy/report/logs

while [ ! -f $LOG/P31_A_DONE ]; do
  pgrep -f "chain_p31.sh" >/dev/null || { echo "p31 gone without marker" > $LOG/P32_ABORT; exit 1; }
  sleep 120
done

for CID in 787bd3b7968b 9f24f74a3fb8 72de269ff79c 83d5553617f1; do
  python -m research.clfstudy.phase --phase 32 --only $CID --max-runs 2 >> $LOG/phase_32.log 2>&1
done
echo "p32 (window A probe): DONE ($(date))" | tee $LOG/P32_A_DONE
