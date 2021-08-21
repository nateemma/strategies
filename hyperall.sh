


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

  # remove hyperopt file (we want the strategy to use the coded values)
  hypfile = "user_data/strategies/${s}.json"
  if [ -f "${hypfile}" ]; then
    rm "${hypfile}"
  fi

  freqtrade hyperopt --space buy --hyperopt-loss OnlyProfitHyperOptLoss --timerange=${timerange} \
    -s $s --no-color --disable-param-export >>hyperoptall.log

done
