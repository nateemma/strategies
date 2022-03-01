#!/bin/zsh

# Run overnight, this will take many hours!

echo ""
echo ""

declare -a elist=("binanceus"  "binance" "ftx" "kucoin")

logfile="temp.log"

zsh user_data/strategies/scripts/download.sh

zsh user_data/strategies/scripts/
for exchange in "${elist[@]}"; do
  logfile="hyp_${exchange}.log"
  echo ""
  echo "=============================="
  echo "${exchange}"
  echo "=============================="
  echo ""

  zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l WeightedProfitHyperOptLoss -s "buy sell" ${exchange}  KalmanSimple
  zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l WeightedProfitHyperOptLoss -s "buy sell" ${exchange}  KalmanSimple2
  zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l WeightedProfitHyperOptLoss -s "buy sell" ${exchange}  FFT
  zsh user_data/strategies/scripts/test_monthly.sh  ${exchange}

  echo ""
  echo "=============================="
done

echo ""
echo ""
