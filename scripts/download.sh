#!/bin/zsh

# exchange list
# make sure kucoin is at the end because it's ...w..a..y...  slower than the others
declare -a list=("binanceus" "binance" "ftx" "kucoin")

run_cmd() {
  cmd="${1}"
  echo "${cmd}"
  eval ${cmd}
}


show_usage () {
    script=$(basename $0)
    cat << END

This script downloads historical data

Usage: zsh $script [options] [<exchange>]

[options]:  -h | --help       print this text
            -l | --leveraged  Use 'leveraged' config file. Optional
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --short      Use 'short' config file. Optional

<exchange>  Name of exchange (binanceus, kucoin, etc). Optional

END
}

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

timerange="${start_date}-"
today=$(date +"%Y%m%d")


#fixed_args="-t 5m 15m 1h 1d"
#fixed_args="-t 5m 15m 1h"
fixed_args="-t 5m"
short=0
leveraged=0


# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts hln:s-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    h | help )       show_usage; exit 0 ;;
    l | leveraged )  leveraged=1 ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; set_start_date; timerange="${start_date}-${today}" ;;
    s | short )      short=1 ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

if [[ $# -gt 0 ]] ; then
  echo "Running for exchange: ${1}"
  list=(${1})
fi

for exchange in "${list[@]}"; do
  echo ""
  echo "${exchange}"
  echo ""

  strat_dir="user_data/strategies/${exchange}"
  config_dir="user_data/strategies/config"
  # config_file="${config_dir}/config_${exchange}.json"
  config_file="${config_dir}/config.json"

  if [ ${short} -eq 1 ]; then
    fixed_args="--trading-mode futures ${fixed_args}"
    config_file="${strat_dir}/config_${exchange}_short.json"
  fi

  if [ ${leveraged} -eq 1 ]; then
    config_file="${strat_dir}/config_${exchange}_leveraged.json"
  fi

  run_cmd "freqtrade download-data  -c ${config_file}  --timerange=${timerange} ${fixed_args}"
  run_cmd "freqtrade download-data  -c ${config_file}  --timerange=${timerange} ${fixed_args} -p BTC/USD BTC/USDT"
done
