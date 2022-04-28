#!/bin/zsh

# Run overnight, this will take many hours!

echo ""
echo ""

#declare -a elist=("binanceus"  "binance" "ftx" "kucoin")
declare -a elist=("binance" "ftx" "kucoin")

declare -a strat_list=("FBB_DWT" "FBB_FFT2" "FBB_Kalman2" "FBB_Kalman2b")


logfile="temp.log"


for exchange in "${elist[@]}"; do
  echo ""
  echo "=============================="
  echo "${exchange}"
  echo "=============================="
  echo ""

  zsh user_data/strategies/scripts/download.sh ${exchange}

  for strat in "${strat_list[@]}"; do
    echo ""
    echo "${strat}"
    echo ""
    cp user_data/strategies/${exchange}/${strat}.json user_data/strategies/${exchange}/${strat}.json.sav
#    zsh user_data/strategies/scripts/hyp_strat.sh -e 2000  -l ExpectancyHyperOptLoss -s "buy sell" ${exchange}  ${strat}
    zsh user_data/strategies/scripts/hyp_strat.sh -e 500  -l ExpectancyHyperOptLoss -s "sell" ${exchange}  ${strat}
  done
  zsh user_data/strategies/scripts/test_monthly.sh  ${exchange}

  echo ""
  echo "=============================="
done

echo ""
echo ""
