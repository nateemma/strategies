#!/bin/bash

# test strategies, run buy parameter hyperopt and compare results for each exchange
# Run overnight, this will take many hours!

declare -a elist=( "binanceus" "ftx" "kucoin" "ascendex" "binance")
#declare -a elist=( "binanceus" )


for exc in "${elist[@]}"; do
  echo "=================="
  echo "$exc"
  echo "=================="
  bash user_data/strategies/scripts/test_exchange.sh $exc

  echo "=============================="
  echo "Hyperopt on $exc"
  echo "=============================="
  bash user_data/strategies/scripts/hyp_exchange.sh $exc

  echo "=========================================="
  echo "Comparing Test Results to Hyperopt Results"
  echo "=========================================="
  bash user_data/strategies/scripts/compareResults $exc
  echo ""
  echo ""
  echo ""
done
