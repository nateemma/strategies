#!/bin/bash

declare -a list=(  "binance" "binanceus" "kucoin" "ascendex" "ftx" )


for exchange in "${list[@]}"; do
  echo ""
  echo "========================"
  echo "    ${exchange}"
  echo "========================"
  echo ""
  bash user_data/strategies/scripts/test_monthly.sh ${exchange}
  echo ""
done


