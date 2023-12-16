#!/bin/zsh

# tests a set of strategies that match a 'group' prefix (e.g. "PCA" or "NNTC_*LSTM")

# list of strategies to test
strat_list=()

# current file (must be at top level)
curr_file="$0"

# default values

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
only_missing_models=false
test_list=${strat_list}
leveraged=0
logfile=""
alt_config=""
run_parallel=true

show_usage () {
#    script=$(basename $ZSH_SOURCE)
    script=${curr_file}
    cat << END

Usage: zsh ${script} [options] <group> <pattern>

[options]:  -c | --config      Specify an alternate config file (just name, not directory or extension)
            -d | --download    Downloads latest market data before running test. Default is ${download}
            -j | --jobs        Number of parallel jobs to run
            -m | --missing     Only run if model is missing
            -n | --ndays       Number of days of backtesting. Defaults to ${num_days}
            -s | --strategy    Test a specific strategy (or list of strategies). Overrides the default list
            -t | --timeframe   Timeframe (YYYMMDD-[YYYMMDD]). Defaults to last ${num_days} days

<group>  Name of group (subdir or exchange)
<pattern>   The pattern to select strategy files. Example: "PCA" will process all strat files beginning with "PCA_"
            TIP: you can also specify the "*" wildcard, but you must enclose this in quotes
                 Example: "NNTC_*LSTM" will test all files that match that pattern.
                 Very useful if you updated a signal type, or a model architecture


END
}


# echo a line to both stdout and the logfile
add_line () {
        echo "${1}" >> $logfile
        echo "${1}"
}

#set -x
# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts c:dj:mn:s:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | config )     needs_arg; alt_config="${OPTARG}" ;;
    d | download )   download=1 ;;
    j | jobs )       needs_arg; jobs="$OPTARG" ;;
    m | missing )    only_missing_models=true ;;
    n | ndays )      needs_arg; num_days="$OPTARG"; set_start_date; timerange="${start_date}-${today}" ;;
    s | strategy )   needs_arg; test_list="$OPTARG"; lev_list="$OPTARG" ;;
    t | timeframe )  needs_arg; timerange="$OPTARG" ;;
    \? )             show_usage; die "Illegal option --$OPT" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list
#set +x


group=$1
pattern=$2

strat_dir="user_data/strategies"
script_dir="${strat_dir}/scripts"
config_dir="${strat_dir}/config"
group_dir="${strat_dir}/${group}"
logfile="test_${group}_${pattern:gs/*/}.log"

if [[ -n ${alt_config} ]]; then
  config_file="${config_dir}/${alt_config}.json"
else
  config_file="${config_dir}/config.json"
fi

if [[ $# -ne 2 ]] ; then
  echo "ERR: Missing arguments"
  show_usage
  exit 0
fi

if [ ! -f ${config_file} ]; then
    echo "config file not found: ${config_file}"
    exit 0
fi

if [ ! -d ${group_dir} ]; then
    echo "Strategy dir not found: ${group_dir}"
    exit 0
fi

# get the list of files using the group identifier

# if there is already a wildcard in the group name, just use that, otherwise use group as a prefix
if [[ ${pattern} == *"*"* ]]; then
  files="${pattern}.py"
else
  files="${pattern}_*.py"
fi

#echo "Matching: ${files}"
find_result=$( find ${group_dir} -name "${files}" -type f -print0 | xargs -0 basename | sort -h )
strat_list=( "${(@f)${find_result}}" )


num_files=${#strat_list[@]}
if [[ $num_files -eq 0 ]]; then
  echo "ERR: no strategy files found for pattern: ${pattern}"
  exit 0
fi


#echo "list: ${list}"
#echo "strat_list: ${strat_list}"

echo ""
echo "Using config file: ${config_file} and Strategy dir: ${group_dir}"
echo ""

# set up path
oldpath=${PYTHONPATH}
export PYTHONPATH="./${group_dir}:./${strat_dir}:${PYTHONPATH}"


# remove any hyperopt files (we want the strategies to use the coded values)
#for entry in $group_dir/*.json
#do
#  echo "removing $entry"
#  rm ${entry}
#done



if [ ${download} -eq 1 ]; then
  echo "Downloading latest data..."
  cmd="freqtrade download-data -t 5m --timerange=${timerange} -c ${config_file}"
  add_line "${cmd}"
  eval ${cmd}
fi

# jarg=""
# if [ ${jobs} -gt 0 ]; then
#     jarg="-j ${jobs}"
# fi

today=`date`
add_line "${today}"

echo "" >$logfile
add_line "Testing strategy list for group: ${group}..."
#add_line "List: ${strat_list}"
add_line "Date/time: ${today}"
add_line "Time range: ${timerange}"
add_line "Log file: ${logfile}"
echo ""

# convert list of files to filtered list of strat names

run_list=( )
for strat in ${strat_list//.py/}; do

  test_strat=true
  if ${only_missing_models}; then
    echo ""
    # model_file="${group_dir}/models/${strat}/${strat}.h5"
    model_file="${group_dir}/models/${strat}/${strat}.keras"
    if [ -e ${model_file} ]; then
      add_line "model file already exists (${model_file}). Skipping strategy ${strat}"
      test_strat=false
    else
      add_line "model file not found (${model_file}). Will train ${strat}"
    fi
  fi

    if ${test_strat}; then
      run_list+=( ${strat} )
    fi
done

# double-check that parallel has been installed
parallel_installed=$(command -v parallel)

if [[ -n $parallel_installed ]]; then
#  echo "parallel is installed"
  echo ""
else
  echo "parallel is not installed. Run brew install parallel (or sudo apt-get install parallel)"
  run_parallel=false
  return
fi

args="${jarg} --timerange=${timerange} -c ${config_file} --strategy-path ${group_dir}"
cmd="freqtrade backtesting --cache none ${args} --strategy-list "

if ${run_parallel}; then
  echo "Running in parallel. Results will be summarised upon completion..."
  echo ""
  echo "Test list:"
  echo ${run_list}
  echo ""
#  echo "parallel -j 3 ${cmd} {} ::: ${run_list} | tee -a ${logfile}"

  parallel -j $jobs -v "${cmd}" {} ::: ${run_list} | tee -a ${logfile}

  wait
  echo $?
else
  for strat in ${run_list}; do
      add_line ""
#      add_line "----------------------"
#      add_line "${strat}"
#      add_line "----------------------"
      command="${cmd} ${strat} >> $logfile"
      add_line "${command}"
      eval ${command}
  done
fi

echo ""
#echo "$logfile:"
#echo ""
#cat $logfile
#echo ""
echo ""

# print a summary of the tests. This also saves the results to the results 'database'
python3 ${script_dir}/SummariseTestResults.py ${logfile}

# restore PYTHONPATH
export PYTHONPATH="${oldpath}"

