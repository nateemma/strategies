#!/bin/bash

declare -a list=("binanceus" "ftx" "kucoin" "ascendex" "binance")

timerange="20210501-"

for exchange in "${list[@]}"; do
  config_file="user_data/strategies/${exchange}/config_${exchange}.json"
    if [ "${exchange}" = "kucoin" ] || [ "${exchange}" = "ascendex" ]; then
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 1m 5m 15m 1h 1d
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 1m 5m 15m 1h 1d -p BTC/USD BTC/USDT
    else
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d
        freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m 15m 1h 1d -p BTC/USD BTC/USDT
    fi
done
