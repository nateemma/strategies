



strat_dir="user_data/strategies"

logfile="hyperall.log"


declare -a elist=("binanceus" "ftx" "kucoin" "ascendex")

timerange="20210501-"

##get date from 120 days ago (MacOS-specific)
#start_date=$(date -j -v-120d +"%Y%m%d")
#timerange="${start_date}-"


echo "" >$logfile
echo "Time range: ${timerange}" >>$logfile
echo "" >>$logfile

for exchange in "${elist[@]}"; do
  echo "" >>$logfile
  echo ============================== >>$logfile
  echo ""
  echo "++++++++++++++++++++++++++++++"
  echo "${exchange}"
  echo "++++++++++++++++++++++++++++++"
  echo ""
  echo "${exchange}" >>$logfile
  echo ============================== >>$logfile
  echo "" >>$logfile


  exchange_dir="user_data/strategies/${exchange}"
  config_file="${exchange_dir}/config_${exchange}.json"

  echo "" >>$logfile
  echo ============================== >>$logfile
  echo "\t\t FisherBBLong" >>$logfile
  echo ============================== >>$logfile
  freqtrade hyperopt --space buy roi stoploss --hyperopt-loss WeightedProfitHyperOptLoss  --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 200 \
    -s FisherBBLong --no-color >>$logfile

  echo "" >>$logfile
  echo ============================== >>$logfile
  echo "\t\t FisherBBQuick" >>$logfile
  echo ============================== >>$logfile
  freqtrade hyperopt --space buy roi stoploss --hyperopt-loss QuickProfitHyperOptLoss  --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 200 \
    -s FisherBBQuick --no-color >>$logfile

  echo "" >>$logfile
  echo ============================== >>$logfile
  echo "\t\t FisherBBExp" >>$logfile
  echo ============================== >>$logfile
  freqtrade hyperopt --space buy roi stoploss --hyperopt-loss ExpectancyHyperOptLoss  --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 200 \
    -s FisherBBExp --no-color >>$logfile

  echo "" >>$logfile
  echo ============================== >>$logfile
  echo "\t\t FisherBBPED" >>$logfile
  echo ============================== >>$logfile
  freqtrade hyperopt --space buy roi stoploss --hyperopt-loss PEDHyperOptLoss  --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 200 \
    -s FisherBBPED --no-color >>$logfile

  echo "" >>$logfile
  echo ============================== >>$logfile
  echo "\t\t FisherBBWinLoss" >>$logfile
  echo ============================== >>$logfile
  freqtrade hyperopt --space buy roi stoploss --hyperopt-loss WinHyperOptLoss  --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 200 \
    -s FisherBBWinLoss --no-color >>$logfile


done

cat $logfile

echo ""
echo ""
echo "Results are in file: ${logfile}"
echo ""

