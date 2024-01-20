#!/bin/zsh

# This script runs hyperopt on all of the main strategies for the specified group and 'pattern'
# 'pattern' is basically the prefix for the files, e.g. "PCA". Any files beginning with the pattern prefix and "_"
# will be processed

# NOTE: for now, only runs with the 'sell' space. May add check to if 'buy' is used later

# Strategy list, and associated hyperopt spaces
#declare -A strat_list=()
strat_list=()

# current file (must be at top level)
curr_file="$0"

# default values
epochs=100
spaces="buy sell"

num_days=180
start_date=$(date +"%Y%m%d")

set_start_date () {
  # ndays="$1"

  # Get the operating system name
  os=$(uname)

  # Check if the operating system is Darwin (macOS)
  if [ "$os" = "Darwin" ]; then
    # Use the -j -v option for BSD date command
    start_date=$(date -j -v-${num_days}d +"%Y%m%d")
  else
    # Use the -d option for GNU date command
    start_date=$(date -d "${num_days} days ago " +"%Y%m%d")
  fi
}

#get date from num_days days ago
set_start_date

today=$(date +"%Y%m%d")
timerange="${start_date}-${today}"
download=0
jobs=0
lossf="ExpectancyHyperOptLoss"
#lossf="SharpeHyperOptLoss"
random_state=$RANDOM
alt_config=""

# get the number of cores
num_cores=`sysctl -n hw.ncpu`
min_cores=$((num_cores - 2))

run_cmd () {
  cmd="${1}"
  echo "${cmd}"
  eval ${cmd}
}


show_usage () {
    script=${curr_file}
    cat << END

Usage: zsh $script [options] <group> <pattern>

[options]:  -c | --config      Specify an alternate config file (just name, not directory or extension)
            -d | --download    Downloads latest market data before running hyperopt. Default is ${download}
            -e | --epochs      Number of epochs to run. Default: ${epochs}
            -j | --jobs        Number of parallel jobs to run
            -l | --loss        Loss function to use (default: ${lossf})
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --spaces      Optimisation spaces (any of: buy, roi, trailing, stoploss, sell). Use quotes for multiple
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days

<group>     Name of group (subdir or exchange)
<pattern>   The prefix of the strategy files. Example: "PCA" will process all strat files beginning with "PCA_"
            TIP: you can also specify the "*" wildcard, but you must enclose this in quotes
                 Example: "NNTC_*LSTM" will test all files that match that pattern.
                 Very useful if you updated a signal type, or a model architecture

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
    add_line "----------------------"
    add_line "${strat}"
    add_line "----------------------"
    add_line ""
    add_line "freqtrade hyperopt ${1}"
    add_line ""
#   set -x
    cmd="freqtrade hyperopt ${1} --no-color >> $logfile"
    eval ${cmd}
#   set +x
}

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }


#-------------------
# Main code
#-------------------

check_shell

while getopts c:d:e:j:l:n:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | config )     needs_arg; alt_config="${OPTARG}" ;;
    d | download )   download=1 ;;
    e | epochs )     needs_arg; epochs="$OPTARG" ;;
    j | jobs )       needs_arg; jobs="$OPTARG" ;;
    l | loss )       needs_arg; lossf="$OPTARG" ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; set_start_date; timerange="${start_date}-${today}" ;;
    s | spaces )     needs_arg; spaces="${OPTARG}" ;;
    t | timeframe )  needs_arg; timerange="$OPTARG" ;;
    \? )             show_usage; die "Illegal option --$OPT" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list


if [[ $# -ne 2 ]] ; then
  echo ""
  echo "ERR: Missing argument(s)"
  echo ""
  show_usage
  exit 0
fi


group=$1
pattern=$2

strat_dir="user_data/strategies"
script_dir="${strat_dir}/scripts"
config_dir="${strat_dir}/config"
group_dir="${strat_dir}/${group}"
logfile="hyp_${group}_${pattern:gs/*/}.log"

if [[ -n ${alt_config} ]]; then
  config_file="${config_dir}/${alt_config}.json"
else
  config_file="${config_dir}/config.json"
fi

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    exit 0
fi

if [ ! -d ${group_dir} ]; then
    echo "Strategy dir not found: ${group_dir}"
    exit 0
fi

# get the list of files using the pattern identifier

# if there is already a wildcard in the pattern name, just use that, otherwise use pattern as a prefix
if [[ ${pattern} == *"*"* ]]; then
  files="${pattern}.py"
else
  files="${pattern}_*.py"
fi

find_result=$( find ${group_dir} -name "${files}" -type f -print0 | xargs -0 basename | sort -h )
strat_list=( "${(@f)${find_result}}" )

num_files=${#strat_list[@]}
if [[ $num_files -eq 0 ]]; then
  echo "ERR: no strategy files found for pattern: ${pattern}"
  exit 0
fi


# calculate min trades
# extract start & end dates from timerange
# a=("${(@s/-/)timerange}")
# start=${a[1]} # don't know why it's reversed
# end=${a[0]}
start=$(echo $timerange | cut -d "-" -f 1)
end=$(echo $timerange | cut -d "-" -f 2)
if [ -z "$end" ]; then
  end="$(date "+%Y%m%d")"
fi
timerange="${start}-${end}"

#echo "timerange:${timerange} start:${start} end:${end}"

# calculate diff
zmodload zsh/datetime
diff=$(( ( $(strftime -r %Y%m%d "$end") - $(strftime -r %Y%m%d "$start") ) / 86400 ))

# set min trades based on # days (N per day)
min_trades=$((diff * 4))


echo "" >$logfile
add_line ""
today=`date`
add_line "============================================="
add_line "Running hyperopt for group: ${group}..."
add_line "Date/time: ${today}"
add_line "Time range: ${timerange}"
add_line "Config file: ${config_file}"
add_line "Strategy dir: ${group_dir}"
add_line ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${group_dir}:./${strat_dir}:${PYTHONPATH}"



if [ ${download} -eq 1 ]; then
    add_line "Downloading latest data..."
    run_cmd "freqtrade download-data  -t 5m --timerange=${timerange} -c ${config_file}"
fi

jarg=""
if [ ${jobs} -gt 0 ]; then
    jarg="-j ${jobs}"
else
  # for kucoin, reduce number of jobs
    if [ "$group" = "kucoin" ]; then
      jarg="-j ${min_cores}"
    fi
fi


hargs=" -c ${config_file} ${jarg} --strategy-path ${group_dir} --timerange=${timerange} --hyperopt-loss ${lossf}"

# add a random state so that each strat starts in the same place
hargs="${hargs} --random-state ${random_state}"

#for strat space in ${(kv)strat_list}; do
for strat in ${strat_list//.py/}; do
  add_line ""


  #  if [[ "${spaces}" == "" ]]; then
  #    spaces=$space
  #  fi

  # run main hyperopt
  args="${hargs} --epochs ${epochs} --space ${spaces} -s ${strat} --min-trades ${min_trades} "
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
