#!/bin/bash

declare -a list=("coinbasepro" "kucoin" "binanceus")
#declare -a list=("kucoin")

logfile="updateROI.log"
echo "" >$logfile

run_hyperopt () {
  exchange=$1
  strategy=$2
  

  # Update buy, roi, and trailing params
  # Note the use of the -j option to save the json files (i.e. cumulative hyperopt)
  # run hyperopt with ShortTradeDurHyperOptLoss for buy, then OnlyProfitHyperOptLoss for other spaces
  echo "============================" >>$logfile
  echo "${strategy} - space: buy" >>$logfile
  echo "" >>$logfile
  bash user_data/strategies/scripts/hypstrat.sh -j -e 100 -l WeightedProfitHyperOptLoss -s "buy" ${exchange} ${strategy} >>$logfile
  echo "${strategy} - space: buy" >>$logfile
  echo "" >>$logfile
  bash user_data/strategies/scripts/hypstrat.sh -j -e 100 -l WeightedProfitHyperOptLoss -s "roi stoploss" ${exchange} ${strategy} >>$logfile
  echo "${strategy} - space: buy" >>$logfile
  echo "" >>$logfile
  bash user_data/strategies/scripts/hypstrat.sh -j -e 100 -l WeightedProfitHyperOptLoss -s "trailing" ${exchange} strategy >>$logfile
  echo "" >>$logfile
  echo "============================" >>$logfile

}


for exchange in "${list[@]}"; do
  run_hyperopt "${exchange}" FisherBBLong
#  run_hyperopt "${exchange}" ComboHold
done

echo ""  >>$logfile
echo "*** Remember to delete the json files ***"  >>$logfile
echo ""  >>$logfile
