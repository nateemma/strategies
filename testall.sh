#!/bin/bash

# list of strategies (only the ones in ComboHold)
slist="ComboHold BBBHold BigDrop BTCBigDrop BTCJump BTCNDrop BTCNSeq EMABounce FisherBB FisherBB2 MACDCross NDrop NSeq "

# remove any hyperopt files (we want the strategies to use the coded values)
for entry in user_data/strategies/*.json
do
  echo "removing $entry"
  rm ${entry}
done

#get date from 30 days ago (MacOS-specific)
start_date=$(date -j -v-30d +"%Y%m%d")

#timerange="20210501-"
#timerange="20210601-"
#timerange="20210701-"
timerange="${start_date}-"

echo "Downloading latest data..."
echo "freqtrade download-data  --timerange=${timerange}"
freqtrade download-data  --timerange=${timerange}

today=`date`
echo $today
echo "Testing strategy list..."
echo "List: ${slist}"
echo "Date/time: ${today}" > Results.log
echo "Time range: ${timerange}" >> Results.log

echo "freqtrade backtesting --timerange=${timerange} --strategy-list ${slist} >> Results.log"
freqtrade backtesting --timerange=${timerange} --strategy-list ${slist} >> Results.log

echo ""
echo "Results.log:"
echo ""
cat Results.log
echo ""
