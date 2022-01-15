#!/bin/bash

# test strategies, run buy parameter hyperopt and compare results for each exchange
# Run overnight, this will take many hours!

#declare -a elist=("binanceus" "ftx" "kucoin" "binance")
declare -a elist=("binance" "ftx" "kucoin")
declare -a slist=("FBB_ROI" "FBB_2" "FBB_2Sqz" "FBB_Solipsis")

echo ""
echo ""

# get the number of cores
num_cores=`sysctl -n hw.ncpu`
min_cores=$((num_cores - 2))

for exc in "${elist[@]}"; do
  echo "=================="
  echo "$exc"
  echo "=================="

  if [ "$exc" = "kucoin" ]; then
    jarg="-j ${min_cores}"
  else
    jarg=""
  fi

  for strat in "${slist[@]}"; do
    if [ "$strat" = "FBB_ROI" ]; then
      spaces="buy roi"
    elif [ "$strat" = "FBB_Solipsis" ]; then
      spaces="buy sell"
    else
      spaces="buy sell roi"
    fi
    echo bash user_data/strategies/scripts/hyp_strat.sh ${jarg} -e 1000 -l SharpeHyperOptLoss -s "\"${spaces}\"" $exc $strat
    bash user_data/strategies/scripts/hyp_strat.sh ${jarg} -e 1000 -l SharpeHyperOptLoss -s "${spaces}" $exc $strat

    if [ "$strat" != "FBB_Solipsis" ]; then
      echo bash user_data/strategies/scripts/hyp_strat.sh ${jarg} -l SharpeHyperOptLoss -e 100 -s "trailing" $exc $strat
      bash user_data/strategies/scripts/hyp_strat.sh ${jarg} -l SharpeHyperOptLoss -e 100 -s trailing $exc $strat
    fi

  done

  echo ""
  echo ""
  echo ""
done
