#!/bin/zsh

# exchange list
# make sure kucoin is at the end because it's ...w..a..y...  slower than the others
declare -a list=("binanceus" "binance" "ftx" "kucoin")

run_cmd() {
  cmd="${1}"
  echo "${cmd}"
  eval ${cmd}
}

#get date from 180 days ago (MacOS-specific)
num_days=180
start_date=$(date -j -v-${num_days}d +"%Y%m%d")
timerange="${start_date}-"


if [[ $# -gt 0 ]] ; then
  echo "Running for exchange: ${1}"
  list=(${1})
fi

for exchange in "${list[@]}"; do
  echo ""
  echo "${exchange}"
  echo ""
#  config_file="user_data/strategies/${exchange}/config_${exchange}_download.json"
  config_file="user_data/strategies/${exchange}/config_${exchange}.json"
  run_cmd "freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d"
  run_cmd "freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d -p BTC/USD BTC/USDT"
done
