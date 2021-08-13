#declare -a list=(
#    "ComboHold" "FisherBB" "NDrop" "BigDrop" "NSeq" "EMABounce" "BBBHold" "BTCJump" "BTCNSeq" "SqueezeOff"
#    "Squeeze002" "TEMABounce" "Hammer" "SARCross" "Squeeze001" "MACDCross" "DonchianBounce" "BBKCBounce"
#    "BuyDips" "SqueezeMomentum" "KeltnerBounce"
#    )

declare -a list=(
    "ComboHold" "FisherBB" "NDrop" "BigDrop" "NSeq" "EMABounce" "BBBHold" "BTCJump" "BTCNSeq" "BTCDrop" "MACDCross"
    )

#freqtrade backtesting --timerange=20210501- --strategy-list "${list}"

timerange="20210701-"

echo "" >hyperoptall.log
echo "Time range: ${timerange}" >> hyperoptall.log
echo "" >>hyperoptall.log

for s in "${list[@]}"; do
  echo ============================== >>hyperoptall.log
  echo ""
  echo "=============================="
  echo "$s"
  echo "=============================="
  echo ""
  echo "$s" >>hyperoptall.log
  echo ============================== >>hyperoptall.log
  freqtrade hyperopt --space buy --hyperopt-loss OnlyProfitHyperOptLoss --timerange=${timerange} \
    -s $s >>hyperoptall.log

done
