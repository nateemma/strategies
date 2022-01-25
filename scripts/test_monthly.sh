#!/usr/local/bin/zsh

# script to run tests over 30 day periods and summarise the results
# makes use of test_exchange.sh
# output is to test_monthly_${exchange}.log

# results arrays
declare -A testResults
declare -a strategies

logfile=""

# func to create the test file
# usage: create_test_file t1 t2, where t1=start days, t2=end days
create_test_file () {

    t1=$1
    t2=$2
    sdate=$(date -j -v-${t1}d +"%Y%m%d")
    edate=$(date -j -v-${t2}d +"%Y%m%d")
    timeframe="${sdate}-${edate}"
    cmd="zsh user_data/strategies/scripts/test_exchange.sh --timeframe=${timeframe} ${exchange}"
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




# Main Code

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    exit 0
fi

exchange=$1

summary_file="test_monthly_${exchange}.log"


echo "" >${summary_file}
echo "              ========================" >>${summary_file}
echo "                  ${exchange}" >>${summary_file}
echo "              ========================" >>${summary_file}
echo "" >>${summary_file}

nperiods=6
dur=30
start=$nperiods*$dur
logfile="test_${exchange}.log"


for (( i=0; i<${nperiods}; i++ )); do

    t1=$(( ${start}-${i}*${dur} ))
    t2=$(( ${t1}-${dur} ))

    create_test_file $t1 $t2

#    sdate=$(date -j -v-${t1}d +"%Y%m%d")
#    edate=$(date -j -v-${t2}d +"%Y%m%d")
#    timeframe="${sdate}-${edate}"
#    echo "zsh user_data/strategies/scripts/test_exchange.sh --timeframe ${timeframe} ${exchange}"
#    zsh user_data/strategies/scripts/test_exchange.sh  --timeframe="${timeframe}" ${exchange}


    # check that files exist
    if [ ! -f ${logfile} ]; then
        echo "test log file not found: ${logfile}"
        exit 0
    fi


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
