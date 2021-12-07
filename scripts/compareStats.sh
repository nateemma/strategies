#!/bin/bash

declare -a list=( "binanceus" "ftx" "kucoin" "ascendex" "binance")

for exchange in "${list[@]}"; do
  echo ""
  echo "              ========================"
  echo "                  ${exchange}"
  echo "              ========================"
  echo ""
  python3  user_data/strategies/scripts/SummariseMonthlyResults.py ./test_monthly_${exchange}.log
  echo ""
done


