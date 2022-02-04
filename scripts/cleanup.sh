#!/bin/zsh

# script to clean up 'old' files

ndays=30
declare -a clean_list=("user_data/backtest_results" "user_data/hyperopt_results" "user_data/plot")

show_usage () {
    script=$(basename $0)
    cat << END

Usage: zsh $script [options]

[options]:  -n | --ndays      Number of days of files to keep
                              any files older than this number of days will be removed. Defaults to ${ndays}

END
}

# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts :n:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    n | ndays )      needs_arg; ndays="$OPTARG" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

if [[ ${ndays} -le 0 ]] ; then
  echo "Must specify no. days > 0"
  exit 0
fi

echo ""
echo "Cleaning files older than ${ndays} days old"
echo ""

for dir in "${clean_list[@]}"; do
echo ""
echo "  Checking directory: ${dir}"
echo ""
  find ${dir} -mtime +${ndays} -type f -exec rm -v {} \;
done

echo ""
