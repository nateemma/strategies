#declare -a list=(
#    "ComboHold" "FisherBB" "NDrop" "BigDrop" "NSeq" "EMABounce" "BBBHold" "BTCJump" "BTCNSeq" "SqueezeOff"
#    "Squeeze002" "TEMABounce" "Hammer" "SARCross" "Squeeze001" "MACDCross" "DonchianBounce" "BBKCBounce"
#    "BuyDips" "SqueezeMomentum" "KeltnerBounce"
#    )

declare -a list=(
  "ComboHold" "BBBHold" "BigDrop" "BTCBigDrop" "BTCJump" "BTCNDrop" "BTCNSeq" "EMABounce" "FisherBB" FisherBB2
  "MACDCross" "NDrop" "NSeq"
)

#get date from 30 days ago (MacOS-specific)
start_date=$(date -j -v-30d +"%Y%m%d")

timerange="${start_date}-"
#timerange="20210701-"

echo "" >hyperoptall.log
echo "Time range: ${timerange}" >>hyperoptall.log
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
    -s $s --no-color >>hyperoptall.log

done
