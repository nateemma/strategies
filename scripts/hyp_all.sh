#!/bin/zsh

# Hyperopt all exchanges

logfile="hyperall.log"


declare -a elist=("binanceus"  "binance" "ftx" "kucoin")


for exchange in "${elist[@]}"; do
  logfile="hyp_${exchange}.log"
  echo ""
  echo "=============================="
  echo "${exchange}"
  echo "=============================="
  echo ""

  zsh user_data/strategies/scripts/hyp_exchange.sh ${exchange}

  cat $logfile
  echo ""
  echo "=============================="
done

echo ""
echo ""


