#!/bin/zsh

# runs hyperopt on a single strategy

show_usage () {
    script=$(basename $BASH_SOURCE)
    cat << END

Usage: zsh $script [options] <exchange> <strategy>

[options]:  -c | --config      path to config file (default: user_data/strategies/<exchange>/config_<exchange>.json
            -e | --epochs      Number of epochs to run. Default 100
            -j | --jobs        Number of parallel jobs
            -l | --loss        Loss function to use (default WeightedProfitHyperOptLoss)
                 --leveraged   Use 'leveraged' config file
            -n | --ndays       Number of days of backtesting. Defaults to 30
            -s | --spaces      Optimisation spaces (any of: buy, roi, trailing, stoploss, sell)
                 --short       Use 'short' config file
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last 30 days

<exchange>  Name of exchange (binanceus, coinbasepro, kucoin, etc)

<strategy>  Name of Strategy

END
}


# Defaults

# loss options: ShortTradeDurHyperOptLoss OnlyProfitHyperOptLoss SharpeHyperOptLoss SharpeHyperOptLossDaily
#               SortinoHyperOptLoss SortinoHyperOptLossDaily

loss="WeightedProfitHyperOptLoss"
#loss="WinHyperOptLoss"

clean=0
epochs=100
jarg=""
config_file=""
short=0
leveraged=0

spaces="sell"

#get date from 180 days ago (MacOS-specific)
num_days=180
start_date=$(date -j -v-${num_days}d +"%Y%m%d")
timerange="${start_date}-"

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts :c:e:j:l:n:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | config )     needs_arg; config_file="$OPTARG" ;;
    e | epochs )     needs_arg; epochs="$OPTARG" ;;
    l | loss )       needs_arg; loss="$OPTARG" ;;
        leveraged )  leveraged=1 ;;
    j | jobs )       needs_arg; jarg="-j $OPTARG" ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; timerange="$(date -j -v-${num_days}d +"%Y%m%d")-" ;;
    s | spaces )     needs_arg; spaces="${OPTARG}" ;;
        short )      short=1 ;;
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
if [ -z "${config_file}" ] ; then
  config_file="${exchange_dir}/config_${exchange}.json"
fi

if [[ $short -ne 0 ]] ; then
    config_file="${exchange_dir}/config_${exchange}_short.json"
fi

if [[ leveraged -ne 0 ]] ; then
    config_file="${exchange_dir}/config_${exchange}_leveraged.json"
fi

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    exit 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    exit 0
fi

# calculate min trades
# extract start & end dates from timerange
a=("${(@s/-/)timerange}")
start=${a[1]} # don't know why it's reversed
end=${a[0]}
if [ -z "$end" ]; then
  end="$(date "+%Y%m%d")"
fi
timerange="${start}-${end}"

#echo "timerange:${timerange} start:${start} end:${end}"

# calculate diff
zmodload zsh/datetime
diff=$(( ( $(strftime -r %Y%m%d "$end") - $(strftime -r %Y%m%d "$start") ) / 86400 ))
min_trades=$((diff / 2))


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


today=`date`
echo $today
echo "Optimising strategy:$strategy for exchange:$exchange..."


#set -x
args="${jarg} --spaces ${spaces} --hyperopt-loss ${loss} --timerange=${timerange} --epochs ${epochs} \
    -c ${config_file} --strategy-path ${exchange_dir}  \
    -s ${strategy} --min-trades ${min_trades} "
cmd="freqtrade hyperopt ${args} --no-color"

cat << END

${cmd}

END
eval ${cmd}
#set +x

#echo -en "\007" # beep
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

