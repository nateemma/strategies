#!/bin/zsh
# Extract & format the BASKET_SUMMARY lines emitted by BasketStrategy at the end
# of each backtest (NAV / cash / income breakdown for every basket strategy).
#
# Usage: zsh basket_summary.sh <logfile>

log="${1:?usage: $0 <logfile>}"
if [[ ! -f "$log" ]]; then
  echo "log file not found: $log" >&2
  exit 1
fi

grep "BASKET_SUMMARY" "$log" | awk '
{
  delete v
  for (i = 1; i <= NF; i++) { split($i, a, "="); v[a[1]] = a[2] }
  printf "%-20s total=%9.2f  ret=%7.2f%%  deployed=%9.2f  cash=%8.2f (avg %4.1f%%)  banked=%8.2f (%.1f%%)\n",
         v["strategy"], v["total"], v["total_ret_pct"], v["deployed"],
         v["cash"], v["avg_cash_pct"], v["banked"], v["banked_pct"]
}'
