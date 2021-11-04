#!/bin/bash

declare -a list=("binanceus" "ftx" "kucoin" "ascendex")

timerange="20210501-"

for exchange in "${list[@]}"; do
  config_file="user_data/strategies/${exchange}/config_${exchange}.json"
  freqtrade download-data  -c ${config_file}  --timerange=${timerange} -t 5m
done
