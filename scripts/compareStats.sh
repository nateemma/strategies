#!/bin/zsh

declare -a list=( "binance" "binanceus" "ftx" "kucoin" )

for exchange in "${list[@]}"; do
  echo "              ========================"
  echo "                  ${exchange}"
  echo "              ========================"
  python3  user_data/strategies/scripts/SummariseMonthlyResults.py ./test_monthly_${exchange}.log
  echo ""
done


