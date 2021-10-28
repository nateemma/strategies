#!/bin/bash

declare -a list=("ftx" "kucoin" "binanceus")

timerange="20210501-"

for exchange in "${list[@]}"; do
  config_file="config_${exchange}.json"
  freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m
  freqtrade download-data  -c ${config_file}  --timerange=${timerange} -p BTC/USD -t 5m
  freqtrade download-data  -c ${config_file}  --timerange=${timerange} -p ETH/USD -t 5m
done
