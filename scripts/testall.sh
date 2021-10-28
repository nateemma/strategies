#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    return 0
fi

exchange=$1
strat_dir="user_data/strategies"
exchange_dir="user_data/strategies/${exchange}"
config_file="config_${exchange}.json"
logfile="testall_${exchange}.log"

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    return 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    return 0
fi

echo ""
echo "Using config file: ${config_file} and Strategy dir: ${exchange_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"

# list of strategies (only the ones in ComboHold)
#slist="ComboHold BBBHold BigDrop BTCBigDrop BTCJump BTCNDrop BTCNSeq EMABounce FisherBB FisherBB2 MACDCross NDrop NSeq FisherBBLong"
slist="FisherBBExp FisherBBLong FisherBBQuick"

# remove any hyperopt files (we want the strategies to use the coded values)
#for entry in $exchange_dir/*.json
#do
#  echo "removing $entry"
#  rm ${entry}
#done

#get date from 120 days ago (MacOS-specific)
start_date=$(date -j -v-120d +"%Y%m%d")

timerange="${start_date}-"

echo "Downloading latest data..."
echo "freqtrade download-data  --timerange=${timerange}"
freqtrade download-data  -t 5m --timerange=${timerange} -c ${config_file}
#freqtrade download-data  -t 1m 5m --timerange=${timerange} -p BTC/USD
#freqtrade download-data  -t 1m 5m 1h --timerange=${timerange} -p ETH/USD

today=`date`
echo $today
echo "Testing strategy list..."
echo "List: ${slist}"
echo "Date/time: ${today}" > $logfile
echo "Time range: ${timerange}" >> $logfile

echo "freqtrade backtesting --timerange=${timerange} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${slist} >> $logfile"
freqtrade backtesting --timerange=${timerange} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${slist} >> $logfile

echo ""
echo "$logfile:"
echo ""
cat $logfile
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

