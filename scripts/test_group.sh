#!/bin/zsh

# tests a set of strategies that match a 'group' prefix (e.g. "PCA")

# list of strategies to test
strat_list=""

# default values

num_days=30
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

Usage: zsh $script [options] <exchange> <group>

[options]:  -d | --download    Downloads latest market data before running test. Default is ${download}
            -j | --jobs        Number of parallel jobs to run
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --strategy    Test a specific strategy (or list of strategies). Overrides the default list
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days

<exchange>  Name of exchange (binanceus, kucoin, etc)
<group>     The prefix of the strategy files. Example: "PCA" will process all strat files beginning with "PCA_"


END
}

#set -x
# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts dj:n:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    d | download )   download=1 ;;
    j | jobs )       needs_arg; jobs="$OPTARG" ;;
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
group=$2

strat_dir="user_data/strategies"
exchange_dir="${strat_dir}/${exchange}"
config_file="${exchange_dir}/config_${exchange}.json"
logfile="test_${exchange}_${group}.log"

if [[ $# -ne 2 ]] ; then
  echo "ERR: Missing arguments"
  show_usage
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

# get the list of files using the group identifier
files="${group}_*.py"
find_result=$( find ${exchange_dir} -name "${files}" -type f -print0 | xargs -0 basename )
list=( "${(@f)${find_result}}" )

num_files=${#list[@]}
if [[ $num_files -eq 0 ]]; then
  echo "ERR: no strategy files found for group: ${group}"
  exit 0
fi

# convert list into space separated string
strat_list=""
for strat in ${list}; do
  strat_list="${strat_list} ${strat//.py/}"
done

#echo "list: ${list}"
#echo "strat_list: ${strat_list}"

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
echo "List: ${strat_list}"
echo "Date/time: ${today}" > $logfile
echo "Time range: ${timerange}" >> $logfile

cmd="freqtrade backtesting  --cache none ${jarg} --timerange=${timerange} -c ${config_file} --strategy-path ${exchange_dir} --strategy-list ${strat_list} > $logfile"
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

