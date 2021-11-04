#!/bin/bash

# runs a single strategy

if [[ $# -lt 2 ]] ; then
  script=$(basename $BASH_SOURCE)
  echo "Usage: bash $script <exchange> <strategy> [<args>]"
  exit 0
fi

exchange=$1
strategy=$2
args=$3

strat_dir="user_data/strategies"
exchange_dir="${strat_dir}/${exchange}"
config_file="${exchange_dir}/config_${exchange}.json"
logfile="test_${exchange}.log"

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    return 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    return 0
fi

echo ""
echo "Using config file: ${config_file} and Strategy dir: ${exchange_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"

# remove any hyperopt files (we want the strategies to use the coded values)
#hypfile="${exchange_dir}/${strategy}.json"
#if [ -f $hypfile ]; then
#  echo "removing $hypfile"
#  rm $hypfile
#fi

#get date from 120 days ago (MacOS-specific)
#start_date=$(date -j -v-120d +"%Y%m%d")
start_date="20210501"

timerange="${start_date}-"

today=`date`
echo $today
echo "Testing strategy:$strategy for exchange:$exchange..."


echo "freqtrade backtesting --timerange=${timerange} ${args} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${strategy}"
freqtrade backtesting --timerange=${timerange} ${args} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${strategy}

echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

