#!/bin/bash

# list of strategies (only the decent ones, in rough order of performance)
slist="ComboHold MFI2 NDrop BigDrop NSeq EMABounce  Strategy003 SqueezeOff BollingerBounce Squeeze002 BBBHold "
slist+="TEMABounce Hammer SARCross Squeeze001  MACDCross DonchianBounce BBKCBounce ADXDM BuyDips "
slist+="SqueezeMomentum Patterns2 KeltnerBounce SimpleBollinger DonchianChannel "
slist+="KeltnerChannels MACDTurn "


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
