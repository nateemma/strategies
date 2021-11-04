#!/bin/bash

declare -a list=( "binanceus" "ftx" "kucoin" "ascendex" )


for exchange in "${list[@]}"; do
  echo ""
  echo "========================"
  echo "    ${exchange}"
  echo "========================"
  echo ""
  bash user_data/strategies/scripts/hyp_exchange.sh -e 100  -s "buy" ${exchange}
  bash user_data/strategies/scripts/hyp_exchange.sh -e 100  -s "roi stoploss" ${exchange}
  bash user_data/strategies/scripts/hyp_exchange.sh -e 100  -s "trailing" ${exchange}
  bash user_data/strategies/scripts/test_exchange.sh ${exchange}
  echo ""
done


