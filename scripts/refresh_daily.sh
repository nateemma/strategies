#!/usr/bin/env bash
# refresh_daily.sh — daily 1d-OHLCV refresh for MomentumRegimeBasket15m's dry-run/live.
#
# WHY: MomentumRegimeBasket15m reads the 1d feathers DIRECTLY off disk (_daily_closes,
# the workaround for freqtrade's startup-candle cap that can't supply the 90d+100d
# history a 15m base needs). A running bot keeps the 15m data live via the DataProvider
# but never touches the 1d feathers, so without this refresh the BTC regime and the
# 90d-momentum reference FREEZE on stale data (ffill holds the last daily close forward)
# and the strategy trades on a frozen signal.
#
# The strategy re-reads the feathers every 15m candle, so it picks up a refresh within
# ~15 min. Run once per day shortly after 00:10 UTC (after the new daily candle closes).
#
# Universe: the static 75-coin config (config_mom_15m.json) — the validated set. The
# dry-run uses a dynamic top-80 VolumePairList; any pick OUTSIDE these 75 won't have a
# 1d feather and so won't be rankable until added to that config's whitelist.
#
# Cron (user crontab; macOS may need Full Disk Access for cron/launchd):
#   10 0 * * * /Users/philprice95/projects/freqtrade/user_data/strategies/scripts/refresh_daily.sh \
#     >> /Users/philprice95/projects/freqtrade/user_data/logs/refresh_daily.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> repo root
exec .venv/bin/freqtrade download-data \
  -c user_data/strategies/config/config_mom_15m.json \
  -t 1d --days 400
