#!/bin/bash

# list of strategies (only the decent ones, in rough order of performance)
slist="ComboHold NDrop NSeq EMABounce  Strategy003  BTCNDrop BTCEMABounce BBBHold Squeeze001 BBKCBounce BuyDips MACDCross "
slist+="KeltnerBounce SimpleBollinger SqueezeOff BollingerBounce Squeeze002 Patterns"


timerange="20210501-"

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
