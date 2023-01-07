#!/bin/zsh

# This script runs hyperopt on all of the main strategies for the specified exchange, using the appropriate
# hyperopt loss function

# Strategy list, and associated hyperopt spaces
declare -A strat_list=( \
[PCA_dwt]="sell" \
[PCA_fbb]="sell" \
[PCA_highlow]="sell" \
[PCA_jump]="sell" \
[PCA_macd]="sell" \
[PCA_mfi]="sell" \
[PCA_minmax]="sell" \
[PCA_nseq]="buy sell" \
[PCA_over]="sell" \
[PCA_profit]="sell" \
[PCA_stochastic]="sell" \
[PCA_swing]="sell" \
)

# default values
epochs=200
spaces=""
num_days=180
start_date=$(date -j -v-${num_days}d +"%Y%m%d")
today=$(date +"%Y%m%d")
timerange="${start_date}-${today}"
download=0
jobs=0
lossf="ExpectancyHyperOptLoss"
#lossf="SharpeHyperOptLoss"
random_state=$RANDOM

# get the number of cores
num_cores=`sysctl -n hw.ncpu`
min_cores=$((num_cores - 2))

run_cmd () {
  cmd="${1}"
  echo "${cmd}"
  eval ${cmd}
}


show_usage () {
    script=$(basename $ZSH_SOURCE)
    cat << END

Usage: zsh $script [options] <exchange>

[options]:  -d | --download    Downloads latest market data before running hyperopt. Default is ${download}
            -e | --epochs      Number of epochs to run. Default: ${epochs}
            -j | --jobs        Number of parallel jobs to run
            -l | --loss        Loss function to use (default: ${lossf})
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --spaces      Optimisation spaces (any of: buy, roi, trailing, stoploss, sell). Use quotes for multiple
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days

<exchange>  Name of exchange (binanceus, ftx, kucoin, etc)


END
}

check_shell () {
  is_zsh= ; : | is_zsh=1
  if [[ "${is_zsh}" != "1" ]]; then
    echo""
    echo "ERR: Must use zsh for this script"
    exit 0
  fi
}

# echo a line to both stdout and the logfile
add_line () {
        echo "${1}" >> $logfile
        echo "${1}"
}

# run the hyperopt command using the supplied arguments ($1)
run_hyperopt () {
      add_line ""
      add_line "freqtrade hyperopt ${1}"
      add_line ""
#      set -x
      cmd="freqtrade hyperopt ${1} --no-color >> $logfile"
      eval ${cmd}
#      set +x
}

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }


#-------------------
# Main code
#-------------------

check_shell

while getopts d:e:j:l:n:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    d | download )   download=1 ;;
    e | epochs )     needs_arg; epochs="$OPTARG" ;;
    j | jobs )       needs_arg; jobs="$OPTARG" ;;
    l | loss )       needs_arg; lossf="$OPTARG" ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; timerange="$(date -j -v-${num_days}d +"%Y%m%d")-" ;;
    s | spaces )     needs_arg; spaces="${OPTARG}" ;;
    t | timeframe )  needs_arg; timerange="$OPTARG" ;;
    \? )             show_usage; die "Illegal option --$OPT" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list


if [[ $# -ne 1 ]] ; then
  echo "ERR: Missing arguments"
  show_usage
  exit 0
fi

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    exit 0
fi

exchange=$1
strat_dir="user_data/strategies"
exchange_dir="${strat_dir}/${exchange}"
config_file="${exchange_dir}/config_${exchange}.json"
logfile="hyp_${exchange}.log"

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    exit 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    exit 0
fi

echo "" >$logfile
add_line ""
today=`date`
add_line "============================================="
add_line "Running hyperopt for exchange: ${exchange}..."
add_line "Date/time: ${today}"
add_line "Time range: ${timerange}"
add_line "Config file: ${config_file}"
add_line "Strategy dir: ${exchange_dir}"
add_line ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"



if [ ${download} -eq 1 ]; then
    add_line "Downloading latest data..."
    run_cmd "freqtrade download-data  -t 5m --timerange=${timerange} -c ${config_file}"
fi

jarg=""
if [ ${jobs} -gt 0 ]; then
    jarg="-j ${jobs}"
else
  # for kucoin, reduce number of jobs
    if [ "$exchange" = "kucoin" ]; then
      jarg="-j ${min_cores}"
    fi
fi


hargs=" -c ${config_file} ${jarg} --strategy-path ${exchange_dir} --timerange=${timerange} --hyperopt-loss ${lossf}"

# add a random state so that each strat starts in the same place
hargs="${hargs} --random-state ${random_state}"

for strat space in ${(kv)strat_list}; do
  add_line ""
  add_line "----------------------"
  add_line "${strat}"
  add_line "----------------------"

  if [[ "${spaces}" == "" ]]; then
    spaces=$space
  fi
  # run main hyperopt
  args="${hargs} --epochs ${epochs} --space ${spaces} -s ${strat}"
  run_hyperopt ${args}

done

echo ""
python user_data/strategies/scripts/SummariseHyperOptResults.py ${logfile}
echo ""


echo ""
echo "Full output is in file: ${logfile}:"
echo ""
#cat $logfile

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"
