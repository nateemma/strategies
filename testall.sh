#!/bin/bash

# list of strategies (only the decent ones, in rough order of performance)
slist="ComboHold FisherBB NDrop BigDrop NSeq EMABounce  Strategy003 BBBHold BTCJump BTCNSeq SqueezeOff "
slist+="Squeeze002 TEMABounce Hammer SARCross Squeeze001  MACDCross DonchianBounce BBKCBounce "
slist+="BuyDips SqueezeMomentum KeltnerBounce "


#timerange="20210501-"
#timerange="20210601-"
timerange="20210701-"

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
