#!/bin/zsh

# Hyperopt all lveraged strats in all (relevant) exchanges

logfile="hyp_leveraged.log"
#loss="SharpeHyperOptLoss"  # weighted versions do not do well with leveraged tokens
loss="ExpectancyHyperOptLoss"
#loss="MedianProfitHyperOptLoss"

declare -a elist=("binance" "ftx" "kucoin")
declare -a slist=("FBB_Leveraged" "FBB_BTCLeveraged" "FBB_BTCJump_Leveraged" "FBB_BTCNSeq_Leveraged")

show_line() {
  echo "${1}"
  echo "${1}" >>${logfile}
}


if [[ $# -gt 0 ]] ; then
  echo "Running for exchange: ${1}"
  elist=(${1})
fi

echo "" >$logfile

for exchange in "${elist[@]}"; do
  show_line ""
  show_line ""
  show_line "++++++++++++++++++++++++++++++"
  show_line "${exchange}"
  show_line "++++++++++++++++++++++++++++++"
  show_line ""

  config="user_data/strategies/${exchange}/config_${exchange}_leveraged.json"

  for strat in "${slist[@]}"; do
    show_line ""
    show_line "=============================="
    show_line "${strat}"
    show_line "=============================="
    show_line ""

    cmd="zsh user_data/strategies/scripts/hyp_strat.sh -e 3000  -l ${loss} -c ${config} ${exchange} ${strat}"
    show_line "${cmd}"
    eval ${cmd} >>$logfile
  done

  show_line ""
  show_line "=============================="
done

echo ""
echo ""
cat $logfile
