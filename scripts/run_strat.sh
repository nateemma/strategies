#!/bin/zsh

# (live) run a  strategy. Takes care of python path, database spec, config file etc.

script=$0

show_usage () {
    cat << END

Usage: zsh $script [options] <group> <strategy>

[options]:  -p | --port       port number (used for naming). Optional
            -l | --leveraged  Use 'leveraged' config file
            -s | --short      Use 'short' config file. Optional

<group>  Either subgroup (e.g. NNTC) or name of group (binanceus, coinbasepro, kucoin, etc)

<strategy>  Name of Strategy

If port is specified, then the script will look for config_<group>_<port>.json

If short is specified, the script will look for config_<group>_short.json

If leverage is specified, the script will look for config_<group>_leverage.json

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

group=$1
strategy=$2

strat_dir="user_data/strategies"
group_dir="${strat_dir}/${group}"
base_config="config.json"
port_config="config_${port}.json"
db_url="tradesv3_${port}.dryrun.sqlite"


exchange_list=$(freqtrade list-exchanges -1)
if [[ "${exchange_list[@]}" =~ $group ]]; then
  echo "Exchange (${group}) detected - using legacy mode"
  base_config="config_${group}.json"
  port_config="config_${group}${port}.json"
  db_url="tradesv3_${group}${port}.dryrun.sqlite"
fi

if [ ${short} -eq 1 ]; then
  # base_config="config_${group}_short.json"
  base_config=$(echo "${base_config}" | sed "s/.json/_short.json/g")
fi

if [[ leveraged -ne 0 ]] ; then
    # base_config="config_${group}_leveraged.json"
    base_config=$(echo "${base_config}" | sed "s/.json/_leveraged.json/g")
fi

if [ ! -f ${base_config} ]; then
    echo "Base config file not found: ${base_config}"
    exit 0
fi

if [ ! -f ${port_config} ]; then
    echo "Port config file not found: ${port_config}"
    exit 0
fi

if [ ! -d ${group_dir} ]; then
    echo "Strategy dir not found: ${group_dir}"
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
echo "Using config file: ${base_config} and Strategy dir: ${group_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${group_dir}:./${strat_dir}:${PYTHONPATH}"

today=`date`
echo $today
echo "Optimising strategy:$strategy for group:$group..."

cmd="freqtrade trade  -c ${config}  --db-url sqlite:///${db_url} --strategy-path ${group_dir} -s ${strategy}"

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

