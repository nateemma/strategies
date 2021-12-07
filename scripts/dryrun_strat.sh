#!/bin/bash

# dry run a  strategy. Takes care of python path, database spec, config file etc.

show_usage () {
    script=$(basename $BASH_SOURCE)
    cat << END

Usage: bash $script [options] <exchange> <strategy>

[options]:  -k | --keep-db   saves the existing database. Removed by default
            -p | --port      port number (used for naming). Optional

<exchange>  Name of exchange (binanceus, kucoin, etc)

<strategy>  Name of Strategy

If port is specified, then the script will look for config_<exchange>_<port>.json

END
}


# Defaults


loss="OnlyProfitHyperOptLoss"
keep_db=0
port=""


timerange="${start_date}-"

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts k:p:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    k | keep-db )    keep_db=1 ;;
    p | port )       needs_arg; port="_$OPTARG" ;;
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
config_file="config_${exchange}${port}.json"
db_url="tradesv3_${exchange}${port}.dryrun.sqlite"

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

if [ ${keep_db} -ne 1 ]; then
  # remove existing database
  if [ -f ${db_url} ]; then
    echo "removing ${db_url}"
    rm ${db_url}
  fi
fi

today=`date`
echo $today
echo "Optimising strategy:$strategy for exchange:$exchange..."

cat << END

-------------------------
freqtrade trade --dry-run -c ${config_file}  --db-url sqlite:///${db_url} --strategy-path ${exchange_dir} -s ${strategy}
-------------------------

END

freqtrade trade --dry-run -c ${config_file}  --db-url sqlite:///${db_url} --strategy-path ${exchange_dir} -s ${strategy}


echo -en "\007" # beep
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

