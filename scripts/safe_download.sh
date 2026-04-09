#!/bin/zsh

# Robust data download script with rate limiting and error handling
# This script handles Binance rate limits and retries failed downloads

declare -a list=("binanceus" "binance" "ftx" "kucoin")

# Configuration
MAX_RETRIES=3
DELAY_BETWEEN_REQUESTS=10  # seconds
DELAY_AFTER_ERROR=60       # seconds after rate limit error

run_cmd_with_retry() {
  local cmd="${1}"
  local exchange="${2}"
  local retry_count=0
  
  while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Executing: ${cmd}"
    echo "Attempt $(($retry_count + 1)) of $MAX_RETRIES"
    
    # Execute the command and capture output
    output=$(eval ${cmd} 2>&1)
    exit_code=$?
    
    # Check for rate limit errors
    if echo "$output" | grep -q "Way too much request weight used\|IP banned\|rate limit\|429\|418"; then
      echo "Rate limit detected for ${exchange}. Waiting ${DELAY_AFTER_ERROR} seconds..."
      sleep $DELAY_AFTER_ERROR
      retry_count=$((retry_count + 1))
      continue
    fi
    
    # Check for other errors
    if [ $exit_code -ne 0 ]; then
      echo "Error occurred (exit code: $exit_code):"
      echo "$output"
      echo "Retrying in ${DELAY_AFTER_ERROR} seconds..."
      sleep $DELAY_AFTER_ERROR
      retry_count=$((retry_count + 1))
      continue
    fi
    
    # Success
    echo "Command executed successfully"
    echo "$output"
    return 0
  done
  
  echo "Failed after $MAX_RETRIES attempts for ${exchange}"
  return 1
}

show_usage () {
    script=$(basename $0)
    cat << END

Robust data download script with rate limiting and error handling

Usage: zsh $script [options] [<exchange>]

[options]:  -h | --help        print this text
            -l | --leveraged   Use 'leveraged' config file. Optional
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --short       Use 'short' config file. Optional
            -t | --timeframe   Timeframe of candles Defaults to ${timeframe}
            -d | --delay       Delay between requests in seconds. Defaults to ${DELAY_BETWEEN_REQUESTS}

<exchange>  Name of exchange (binanceus, kucoin, etc). Optional

END
}

num_days=180
start_date=$(date +"%Y%m%d")

set_start_date () {
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
timeframe='5m'

short=0
leveraged=0

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts hln:st:d:-: OPT; do
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
    t | timeframe )  timeframe=${OPTARG} ;;
    d | delay )      needs_arg; DELAY_BETWEEN_REQUESTS="$OPTARG" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

fixed_args="-t ${timeframe}"

if [[ $# -gt 0 ]] ; then
  echo "Running for exchange: ${1}"
  list=(${1})
fi

echo "Starting robust data download with ${DELAY_BETWEEN_REQUESTS}s delays between requests"
echo "Timerange: ${timerange}"
echo "Timeframe: ${timeframe}"
echo ""

for exchange in "${list[@]}"; do
  echo ""
  echo "=========================================="
  echo "Processing exchange: ${exchange}"
  echo "=========================================="
  echo ""

  strat_dir="user_data/strategies/${exchange}"
  config_dir="user_data/strategies/config"
  config_file="${config_dir}/config.json"

  if [ ${short} -eq 1 ]; then
    fixed_args="--trading-mode futures ${fixed_args}"
    config_file="${strat_dir}/config_${exchange}_short.json"
  fi

  if [ ${leveraged} -eq 1 ]; then
    config_file="${strat_dir}/config_${exchange}_leveraged.json"
  fi

  # Download all pairs
  echo "Downloading all pairs for ${exchange}..."
  cmd="freqtrade download-data -c ${config_file} --timerange=${timerange} ${fixed_args}"
  run_cmd_with_retry "$cmd" "$exchange"
  
  # Wait between requests
  echo "Waiting ${DELAY_BETWEEN_REQUESTS} seconds before next request..."
  sleep $DELAY_BETWEEN_REQUESTS
  
  # Download specific pairs
  echo "Downloading specific pairs for ${exchange}..."
  cmd="freqtrade download-data -c ${config_file} --timerange=${timerange} ${fixed_args} -p BTC/USD BTC/USDT"
  run_cmd_with_retry "$cmd" "$exchange"
  
  # Wait between exchanges
  if [ "$exchange" != "${list[-1]}" ]; then
    echo "Waiting ${DELAY_BETWEEN_REQUESTS} seconds before next exchange..."
    sleep $DELAY_BETWEEN_REQUESTS
  fi
done

echo ""
echo "Data download completed!" 