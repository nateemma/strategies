#!/bin/bash

# runs hyperopt on a single strategy

show_usage () {
    script=$(basename $BASH_SOURCE)
    cat << END

Usage: bash $script [options] <exchange> <strategy>

[options]:  -c | --clean       Remove hyperopt .json file before running
            -e | --epochs      Number of epochs to run. Default 100
            -j | --save-json   Saves hyperopt json file (for cumulative runs)
            -l | --loss        { ShortTradeDurHyperOptLoss | OnlyProfitHyperOptLoss | SharpeHyperOptLoss |
                               SharpeHyperOptLossDaily | SortinoHyperOptLoss | SortinoHyperOptLossDaily }
            -n | --ndays       Number of days of backtesting. Defaults to 30
            -s | --spaces      Optimisation spaces (any of: buy, roi, trailing, stoploss, sell)
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last 30 days

<exchange>  Name of exchange (binanceus, coinbasepro, kucoin, etc)

<strategy>  Name of Strategy

END
}


# Defaults

# loss options: ShortTradeDurHyperOptLoss OnlyProfitHyperOptLoss SharpeHyperOptLoss SharpeHyperOptLossDaily
#               SortinoHyperOptLoss SortinoHyperOptLossDaily
loss="OnlyProfitHyperOptLoss"

export="--disable-param-export"
save_json=1
clean=0
epochs=100

spaces="buy"

#get date from 30 days ago (MacOS-specific)
num_days=30
#start_date=$(date -j -v-30d +"%Y%m%d")
start_date="20210501"
timerange="${start_date}-"

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts :ce:jl:n:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | clean )      clean=1 ;;
    e | epochs )     needs_arg; epochs="$OPTARG" ;;
    l | loss )       needs_arg; loss="$OPTARG" ;;
    j | save-json )  save_json=1 ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; timerange="$(date -j -v-${num_days}d +"%Y%m%d")-" ;;
    s | spaces )     needs_arg; spaces="$OPTARG" ;;
    t | timeframe )  needs_arg; timerange="$OPTARG" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list


if [[ $# -ne 2 ]] ; then
  echo "ERR: Missing arguments"
  show_usage
  exit 0
fi

exchange=$1
strategy=$2

strat_dir="user_data/strategies"
exchange_dir="${strat_dir}/${exchange}"
config_file="${exchange_dir}/config_${exchange}.json"

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    exit 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    exit 0
fi

echo ""
echo "Using config file: ${config_file} and Strategy dir: ${exchange_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"

hypfile="${exchange_dir}/${strategy}.json"

if [ ${clean} -eq 1 ]; then
  # remove any hyperopt files (we want the strategies to use the coded values)
  if [ -f $hypfile ]; then
    echo "removing $hypfile"
    rm $hypfile
  fi
fi

if [ ${save_json} -eq 1 ]; then
  # save output.json file
  echo "hyperopt output file (${hypfile} will be saved"
  export=""
fi

today=`date`
echo $today
echo "Optimising strategy:$strategy for exchange:$exchange..."

cat << END

freqtrade hyperopt -j 6 --space ${spaces} --hyperopt-loss ${loss} --timerange=${timerange} --epochs ${epochs} \
-c ${config_file} --strategy-path ${exchange_dir}  \
-s ${strategy} --no-color ${export}


END

freqtrade hyperopt  -j 6 --space ${spaces} --hyperopt-loss ${loss} --timerange=${timerange} --epochs ${epochs} \
    -c ${config_file} --strategy-path ${exchange_dir}  \
    -s ${strategy} --no-color ${export}

echo -en "\007" # beep
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

