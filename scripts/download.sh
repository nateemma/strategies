#!/bin/bash

# exchange list
# make sure kucoin is at the end because it's ...w..a..y...  slower than the others
declare -a list=("binanceus" "binance" "ftx" "kucoin")

timerange="20210501-"

for exchange in "${list[@]}"; do
  echo ""
  echo "${exchange}"
  echo ""
  config_file="user_data/strategies/${exchange}/config_${exchange}.json"
    if [ "${exchange}" = "kucoin" ] || [ "${exchange}" = "ascendex" ]; then
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 1m 5m 15m 1h 1d
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 1m 5m 15m 1h 1d -p BTC/USD BTC/USDT
    else
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d -p BTC/USD BTC/USDT
    fi
done
