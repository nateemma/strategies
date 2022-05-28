#!/usr/local/bin/zsh

# script to run tests over 30 day periods and summarise the results
# makes use of test_exchange.sh
# output is to test_monthly_${exchange}.log

# results arrays
declare -A testResults
declare -a strategies

logfile=""
leveraged=0
lev_arg=""
strat_arg=""

show_usage () {
    script=$(basename $0)
    cat << END

Usage: zsh $script [options] <exchange>

[options]:  -l | --leveraged   Test leveraged strategies
            -s | --strategy    Test a specific strategy (or list of strategies in quotes). Overrides the default list

<exchange>  Name of exchange (binanceus, ftx, kucoin, etc)


END
}


# func to create the test file
# usage: create_test_file t1 t2, where t1=start days, t2=end days
create_test_file () {

    t1=$1
    t2=$2
    sdate=$(date -j -v-${t1}d +"%Y%m%d")
    edate=$(date -j -v-${t2}d +"%Y%m%d")
    timeframe="${sdate}-${edate}"
    cmd="zsh user_data/strategies/scripts/test_exchange.sh ${lev_arg} ${strat_arg} --timeframe=${timeframe} ${exchange}"
    echo "${cmd}"
    eval ${cmd}

}


# func to parse the test file
scan_test_file () {
  # states
  st_find_summary=0
  st_header=1
  st_results=2
  st_end=3
  state=$st_find_summary

  strategy=""
  result=""
  count=1


  # check file
  if [ -f "${logfile}" ]; then
    # loop through lines
    while IFS='' read -r line || [[ -n "${line}" ]]; do
      if  [ $state -eq $st_find_summary ]; then
        # scan until summary table found
        if [[ $line =~ .*"Max open trades :".* ]]; then
          echo ""
          echo "${line}"
          state=$st_header
          count=2
        fi

      elif [ $state -eq $st_header ]; then
        # read $count lines
        if [ $count -gt 0 ]; then
          count=$((count - 1))
          echo "${line}"
       else
          state=$st_results
        fi

      elif [ $state -eq $st_results ]; then
        echo "${line}"

        # process results until end of table found
        if [[ $line =~ .*"==============".* ]]; then
          echo "" >> $logfile
          state=$st_end
        else
          # extract strategy name and results
          items=(${line//|/ })
          strategy=${items[0]}
          result=${items[5]}
          #echo "${line}"
          #echo "Test: ${strategy}: ${result}"
          testResults[${strategy}]+=${result}
          strategies+=($strategy) # maintains order of insertion
        fi

      elif [ $state -eq $st_end ]; then
        # do nothing
        return

      else
        echo "ERROR: invalid state: ${state}"
      fi
    done < "${logfile}"
    #echo "#=========================================="

  else
    echo "ERROR: ${logfile} not found"
  fi

}


#set -x
# process options
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts ls:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    l | leveraged )  leveraged=1;;
    s | strategy )   needs_arg; strat_arg="-s \"$OPTARG\"" ;;
    \? )             show_usage; die "Illegal option --$OPT" ;;
    ??* )            show_usage; die "Illegal option --$OPT" ;;  # bad long option
    ? )              show_usage; die "Illegal option --$OPT" ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list
#set +x


# Main Code

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    exit 0
fi

exchange=$1
nperiods=6
dur=30
start=$nperiods*$dur

summary_file="test_monthly_${exchange}.log"
logfile="test_${exchange}.log"

if [ ${leveraged} -eq 1 ]; then
  lev_arg="--leveraged"
  logfile="test_leveraged_${exchange}.log"
  summary_file="test_monthly_leveraged_${exchange}.log"
fi


echo "" >${summary_file}
echo "              ========================" >>${summary_file}
echo "                  ${exchange}" >>${summary_file}
echo "              ========================" >>${summary_file}
echo "" >>${summary_file}



for (( i=0; i<${nperiods}; i++ )); do

    t1=$(( ${start}-${i}*${dur} ))
    t2=$(( ${t1}-${dur} ))

    create_test_file $t1 $t2

    # check that test iles exists
    if [ ! -f ${logfile} ]; then
        echo "test log file not found: ${logfile}"
        exit 0
    fi

    # scan the test results and append to the summary file
    scan_test_file >>${summary_file}

done



echo "" >>${summary_file}
echo "Overall Statistics:" >>${summary_file}
echo "" >>${summary_file}
python3 user_data/strategies/scripts/SummariseMonthlyResults.py ${summary_file} >>${summary_file}


echo ""
echo ""
cat ${summary_file}

echo ""
echo ""
