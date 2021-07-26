declare -a list=(
  "ComboHold" "MFI2" "NDrop" "BigDrop" "NSeq" "EMABounce" "Strategy003" "BBBHold" "Patterns2" "Squeeze001"
  "ADXDM" "BBKCBounce" "BuyDips" "MACDCross" "DonchianBounce" "TEMABounce" "Hammer"
  "KeltnerBounce" "SimpleBollinger" "SqueezeOff" "BollingerBounce" "Squeeze002" "DonchianChannel"
  "SqueezeMomentum" "KeltnerChannels" "MACDTurn" "SARCross"
  )

#freqtrade backtesting --timerange=20210501- --strategy-list "${list}"

echo "" >hyperoptall.log
for s in "${list[@]}"; do
  echo ============================== >>hyperoptall.log
  echo ""
  echo "$s"
  echo ""
  echo "$s" >>hyperoptall.log
  echo ============================== >>hyperoptall.log
  freqtrade hyperopt --space buy --hyperopt-loss OnlyProfitHyperOptLoss --timerange=20210601- \
    -s $s >>hyperoptall.log

done
