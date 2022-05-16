#!/bin/zsh

# (live) run a  strategy. Takes care of python path, database spec, config file etc.

show_usage () {
    script=$(basename $0)
    cat << END

Usage: zsh $script [options] <exchange> <strategy>

[options]:  -p | --port      port number (used for naming). Optional
            -l | --leveraged   Use 'leveraged' config file
            -s | --short     Use 'short' config file. Optional

<exchange>  Name of exchange (binanceus, kucoin, etc)

<strategy>  Name of Strategy

If port is specified, then the script will look for config_<exchange>_<port>.json

If short is specified, the script will look for config_<exchange>_short.json

If leverage is specified, the script will look for config_<exchange>_leverage.json

NOTE: if the database already exists, it will be re-used, i.e. any previously opened trades should be found

END
}


# Defaults

port=""
short=0
leveraged=0


timerange="${start_date}-"

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts lp:s-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    l | leveraged )  leveraged=1 ;;
    p | port )       needs_arg; port="_$OPTARG" ;;
    s | short )      short=1 ;;
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
base_config="config_${exchange}.json"
port_config="config_${exchange}${port}.json"
db_url="tradesv3_${exchange}${port}.sqlite"

if [ ${short} -eq 1 ]; then
  base_config="config_${exchange}_short.json"
fi

if [[ leveraged -ne 0 ]] ; then
    base_config="config_${exchange}_leveraged.json"
fi

if [ ! -f ${base_config} ]; then
    echo "Base config file not found: ${base_config}"
    exit 0
fi

if [ ! -f ${port_config} ]; then
    echo "Port config file not found: ${port_config}"
    exit 0
fi

if [ ! -d ${exchange_dir} ]; then
    echo "Strategy dir not found: ${exchange_dir}"
    exit 0
fi

if [ -f ${db_url} ]; then
    echo "Re-using database: ${db_url}"
fi

# set up config file chain (if port specified)
config="${base_config}"
if [[ ${port} == "" ]]; then
  config="${base_config}"
else
  config="${base_config} -c ${port_config}"
fi

echo ""
echo "Using config file: ${base_config} and Strategy dir: ${exchange_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${exchange_dir}:./${strat_dir}:${PYTHONPATH}"

today=`date`
echo $today
echo "Optimising strategy:$strategy for exchange:$exchange..."

cmd="freqtrade trade  -c ${config}  --db-url sqlite:///${db_url} --strategy-path ${exchange_dir} -s ${strategy}"

cat << END

LIVE RUN!

-------------------------
${cmd}
-------------------------

END


eval ${cmd}

echo -en "\007" # beep
echo ""

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

