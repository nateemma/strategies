#!/bin/bash

# list of strategies (only the decent ones, in rough order of performance)
slist="ComboHold NDrop BigDrop NSeq EMABounce  Strategy003 MFI2 SqueezeOff BollingerBounce BBBHold Squeeze001  "
slist+="MACDCross DonchianBounce TEMABounce BBKCBounce BuyDips Hammer SARCross ADXDM "
slist+="SqueezeMomentum Patterns2 KeltnerBounce SimpleBollinger Squeeze002 DonchianChannel "
slist+="KeltnerChannels MACDTurn MFIRSICross "


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
