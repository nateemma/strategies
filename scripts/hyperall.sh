


if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    return 0
fi

exchange=$1
strat_dir="user_data/strategies"
exchange_dir="user_data/strategies/${exchange}"
config_file="config_${exchange}.json"
logfile="hyperall_${exchange}.log"

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    return 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    return 0
fi


# set up path (needed to resolve imports)
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"

echo ""
echo "Using config file: ${config_file} and Strategy dir: ${exchange_dir}"
echo ""

#declare -a list=(
#  "ComboHold" "BBBHold" "BigDrop" "BTCBigDrop" "BTCJump" "BTCNDrop" "BTCNSeq" "EMABounce" "FisherBB" "FisherBB2"
#  "MACDCross" "NDrop" "NSeq"
#)
declare -a list=(
  "FisherBB2" "MACDCross" "NDrop" "NSeq"
)

#get date from 120 days ago (MacOS-specific)
start_date=$(date -j -v-120d +"%Y%m%d")

timerange="${start_date}-"
#timerange="20210701-"

echo "" >$logfile
echo "Time range: ${timerange}" >>$logfile
echo "" >>$logfile

for s in "${list[@]}"; do
  echo ============================== >>$logfile
  echo ""
  echo "=============================="
  echo "$s"
  echo "=============================="
  echo ""
  echo "$s" >>$logfile
  echo ============================== >>$logfile

  # remove hyperopt file (we want the strategy to use the coded values)
  hypfile="${exchange_dir}/${s}.json"
  if [ -f "${hypfile}" ]; then
    rm "${hypfile}"
  fi

  # loss options: ShortTradeDurHyperOptLoss OnlyProfitHyperOptLoss SharpeHyperOptLoss SharpeHyperOptLossDaily
  #               SortinoHyperOptLoss SortinoHyperOptLossDaily
  loss="SharpeHyperOptLoss"
  freqtrade hyperopt -j 6 --space buy --hyperopt-loss ${loss} --timerange=${timerange} \
    -c ${config_file} --strategy-path ${exchange_dir}  --epochs 300 \
    -s $s --no-color --disable-param-export >>$logfile

done
