#!/bin/zsh

declare -a list=(  "binance" "binanceus" "ftx" "kucoin" )


for exchange in "${list[@]}"; do
  echo ""
  echo "========================"
  echo "    ${exchange}"
  echo "========================"
  echo ""
  zsh user_data/strategies/scripts/test_monthly.sh ${exchange}
  echo ""
done

zsh user_data/strategies/scripts/compareStats.sh


