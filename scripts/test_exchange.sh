#!/bin/zsh

# list of strategies to test
lev_list="DWT_Leveraged DWT_lev_short DWT_Leveraged_recent DWT_Leveraged2"
strat_list="PCA_dwt PCA_fbb PCA_highlow PCA_jump PCA_macd PCA_mfi PCA_minmax PCA_nseq PCA_over PCA_profit \
PCA_stochastic PCA_swing"

# default values

num_days=180
start_date=$(date -j -v-${num_days}d +"%Y%m%d")
today=$(date +"%Y%m%d")
timerange="${start_date}-${today}"
download=0
jobs=0
test_list=${strat_list}
leveraged=0

show_usage () {
    script=$(basename $ZSH_SOURCE)
    cat << END

Usage: zsh $script [options] <exchange>

[options]:  -d | --download    Downloads latest market data before running hyperopt. Default is ${download}
            -j | --jobs        Number of parallel jobs to run
            -l | --leveraged   Test leveraged strategies
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --strategy    Test a specific strategy (or list of strategies). Overrides the default list
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days

<exchange>  Name of exchange (binanceus, ftx, kucoin, etc)


END
}

#set -x
# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts dj:ln:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    d | download )   download=1 ;;
    j | jobs )       needs_arg; jobs="$OPTARG" ;;
    l | leveraged )  leveraged=1;;
    n | ndays )      needs_arg; num_days="$OPTARG"; timerange="$(date -j -v-${num_days}d +"%Y%m%d")-${today}" ;;
    s | strategy )   needs_arg; test_list="$OPTARG"; lev_list="$OPTARG" ;;
    t | timeframe )  needs_arg; timerange="$OPTARG" ;;
    \? )             show_usage; die "Illegal option --$OPT" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list
#set +x


exchange=$1
strat_dir="user_data/strategies"
exchange_dir="${strat_dir}/${exchange}"
config_file="${exchange_dir}/config_${exchange}.json"
logfile="test_${exchange}.log"

if [ ${leveraged} -eq 1 ]; then
  test_list=${lev_list}
  config_file="${exchange_dir}/config_${exchange}_leveraged.json"
  logfile="test_leveraged_${exchange}.log"
fi

if [[ $# -ne 1 ]] ; then
  echo "ERR: Missing arguments"
  show_usage
  exit 0
fi

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    exit 0
fi

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


# remove any hyperopt files (we want the strategies to use the coded values)
#for entry in $exchange_dir/*.json
#do
#  echo "removing $entry"
#  rm ${entry}
#done



if [ ${download} -eq 1 ]; then
  echo "Downloading latest data..."
  cmd="freqtrade download-data -t 5m --timerange=${timerange} -c ${config_file}"
  echo "${cmd}"
  eval ${cmd}
fi

jarg=""
if [ ${jobs} -gt 0 ]; then
    jarg="-j ${jobs}"
fi

today=`date`
echo "${today}"

echo "Testing strategy list for exchange: ${exchange}..."
echo "List: ${slist}"
echo "Date/time: ${today}" > $logfile
echo "Time range: ${timerange}" >> $logfile

cmd="freqtrade backtesting  --cache none ${jarg} --timerange=${timerange} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${test_list} > $logfile"
echo "${cmd}"
eval ${cmd}

echo ""
echo "$logfile:"
echo ""
cat $logfile
echo ""
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

