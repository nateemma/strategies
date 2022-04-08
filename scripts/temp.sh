#!/bin/zsh

# Run overnight, this will take many hours!

echo ""
echo ""

#declare -a elist=("binanceus"  "binance" "ftx" "kucoin")
declare -a elist=("ftx")

declare -a strat_list=("FBB_DWT" "FBB_DWT2" "FBB_Kalman2" "FBB_KalmanSIMD")


logfile="temp.log"

zsh user_data/strategies/scripts/download.sh

for exchange in "${elist[@]}"; do
  echo ""
  echo "=============================="
  echo "${exchange}"
  echo "=============================="
  echo ""

  for strat in "${strat_list[@]}"; do
    echo ""
    echo "${strat}"
    echo ""
    zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l ExpectancyHyperOptLoss -s "buy" ${exchange}  ${strat}
    zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l ExpectancyHyperOptLoss -s "sell" ${exchange}  ${strat}
    zsh user_data/strategies/scripts/hyp_strat.sh -e 1000  -l ExpectancyHyperOptLoss -s "buy" ${exchange}  ${strat}
  done
  zsh user_data/strategies/scripts/test_monthly.sh  ${exchange}

  echo ""
  echo "=============================="
done

echo ""
echo ""
