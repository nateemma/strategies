#!/bin/zsh

# runs a single strategy

script=$0

clean=0
epochs=100
jarg=""
config_file=""
short=0
leveraged=0

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


end_date="$(date "+%Y%m%d")"

timerange="${start_date}-${end_date}"

default_config="user_data/strategies/config/config.json"

show_usage () {
    cat << END

Usage: zsh $script [options] <group> <strategy>

[options]:  -c | --config      path to config file (default: ${default_config}
            -l | --leveraged   Use 'leveraged' config file
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
                 --short       Use 'short' config file
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days (${timerange})

<group>  Either subgroup (e.g. NNTC) or name of exchange (binanceus, coinbasepro, kucoin, etc)

<strategy>  Name of Strategy

Note that if an exchange is specified, the config file can also be user_data/strategies/<exchange>/config_<exchange>.json
This is for backwards compatibility. Be aware that the script will check for ${default_config} first

END
}

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts :c:e:j:ln:st:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | config )     needs_arg; config_file="$OPTARG" ;;
    e | epochs )     needs_arg; epochs="$OPTARG" ;;
    l | leveraged )  leveraged=1 ;;
    j | jobs )       needs_arg; jarg="-j $OPTARG" ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; set_start_date; timerange="${start_date}-${today}" ;;
    s | short )      short=1 ;;
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

group=$1
strategy=$2

strat_dir="user_data/strategies"
config_dir="${strat_dir}/config"
group_dir="${strat_dir}/${group}"
strat_file="${group_dir}/${strategy}.py"
utils_dir="${strat_dir/utils}"

exchange_list=$(freqtrade list-exchanges -1)
if [[ "${exchange_list[@]}" =~ $group ]]; then
  echo "Exchange (${group}) detected - using legacy mode"
  exchange="_${group}"
  config_dir="${group_dir}"
else
  exchange=""
fi


if [[ $short -ne 0 ]] ; then
    config_file="${config_dir}/config${exchange}_short.json"
fi

if [[ leveraged -ne 0 ]] ; then
    config_file="${config_dir}/config${exchange}_leveraged.json"
fi

if [ -z "${config_file}" ] ; then
  config_file="${config_dir}/config${exchange}.json"
fi

if [ ! -f ${config_file} ]; then
    echo ""
    echo "config file not found: ${config_file}"
    echo "(Maybe try using the -c option?)"
    echo ""
    exit 0
fi

if [ ! -d ${group_dir} ]; then
    echo ""
    echo "Strategy dir not found: ${group_dir}"
    echo ""
    exit 0
fi

if [ ! -f  ${strat_file} ]; then
    echo "Strategy file file not found: ${strat_file}"
    exit 0
fi


# adjust timerange to make sure there is an end date (which enables caching of data in backtesting)
# a=("${(@s/-/)timerange}")
# start=${a[1]} # don't know why it's reversed
# end=${a[0]}
start=$(echo $timerange | cut -d "-" -f 1)
end=$(echo $timerange | cut -d "-" -f 2)
if [ -z "$end" ]; then
  end="$(date "+%Y%m%d")"
fi
timerange="${start}-${end}"


echo ""
echo "Using config file: ${config_file} and Strategy dir: ${group_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${group_dir}:./${strat_dir}:./${utils_dir}:${PYTHONPATH}"

today=`date`
echo $today
echo "Testing strategy:$strategy for exchange:$exchange..."


cmd="freqtrade backtesting --cache none  --breakdown month --timerange=${timerange} -c ${config_file} --strategy-path ${group_dir} --strategy-list ${strategy}"
#cmd="freqtrade backtesting --breakdown month --timerange=${timerange} -c ${config_file} --strategy-path ${group_dir} --strategy-list ${strategy}"
echo ${cmd}
eval ${cmd}

echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"