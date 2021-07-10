declare -a list=("ADXDM" "BBBHold" "BollingerBounce" "BuyDips" "DonchianBounce" "DonchianChannel"
  "EMA003" "EMACross" "KeltnerBounce" "KeltnerChannels" "MACD003" "MACDCross" "MFI2"
  "SimpleBollinger" "Squeeze001" "Squeeze002" "SqueezeMomentum" "SqueezeOff" "Strategy003"
)

#freqtrade backtesting --timerange=20210501- --strategy-list "${list}"

echo "" >hyperoptall.log
for s in "${list[@]}"; do
  echo ============================== >>hyperoptall.log
  echo "$s"
  echo "$s" >>hyperoptall.log
  echo ============================== >>hyperoptall.log
  freqtrade hyperopt --space roi stoploss trailing --hyperopt-loss SharpeHyperOptLossDaily --timerange=20210501- \
    -s $s >>hyperoptall.log

done
