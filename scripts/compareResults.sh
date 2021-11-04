#!/usr/local/bin/bash

# script to scan the test results and hyperopt results and provide a comparison
# output is to stdout

# results arrays
declare -A testResults
declare -A hyperResults
declare -A hyperParams
declare -a strategies

testfile=""
hyperfile=""

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

#  set -x
  set +x

  # check file
  if [ -f "${testfile}" ]; then
    # loop through lines
    while IFS='' read -r line || [[ -n "${line}" ]]; do
      if  [ $state -eq $st_find_summary ]; then
        # scan until summary table found
        if [[ $line =~ .*"STRATEGY SUMMARY".* ]]; then
          state=$st_header
          count=1
        fi

      elif [ $state -eq $st_header ]; then
        # read $count lines
        if [ $count -gt 0 ]; then
          count=$((count - 1))
        else
          state=$st_results
        fi

      elif [ $state -eq $st_results ]; then
        # process results until end of table found
        if [[ $line =~ .*"==============".* ]]; then
          state=$st_end
        else
          # extract strategy name and results
          items=(${line//|/ })
          strategy=${items[0]}
          result=${items[5]}
          #echo "${line}"
          #echo "Test: ${strategy}: ${result}"
          testResults[${strategy}]="${result}"
          strategies+=($strategy) # maintains order of insertion
        fi

      elif [ $state -eq $st_end ]; then
        # do nothing
        return

      else
        echo "ERROR: invalid state: ${state}"
      fi
    done < "${testfile}"
    #echo "#=========================================="

  else
    echo "ERROR: ${testfile} not found"
  fi

  set +x
}

# func to parse the hyperopt file
scan_hyper_file () {
    
  # states
  st_find_name=0
  st_name=1
  st_find_results=2
  st_find_params=3
  st_params=4

  state=$st_find_name
  strategy=""
  strat_params=""
  count=0
  
  #set -x
  set +x
  
  # check file
  if [ -f "${hyperfile}" ]; then
    # loop through lines
    while IFS='' read -r line || [[ -n "${line}" ]]; do
      if  [ $state -eq $st_find_name ]; then
        # scan until separator found
        if [[ $line =~ .*"==========".* ]]; then
          state=$st_name
        fi
  
      elif [ $state -eq $st_name ]; then
        # extract strategy name
        strategy=$line
        #echo "# ${strategy}"
        state=$st_find_results
  
      elif [ $state -eq $st_find_results ]; then
        if [[ $line =~ .*"No epochs evaluated yet".* ]]; then
          echo "# No parameters found for $strategy!"
          echo ""
          hyperResults[${strategy}]="0.00"
          state=$st_find_name
        fi
        # scan until "Total profit" found
        if [[ $line =~ .*"Total profit".* ]]; then
          #echo "$line"
          # extract the profit number
          tmp=${line#*(}   # remove up to "("
          result=${tmp%)*}   # remove from ")"
          result="${result// }"
          result="${result//%}"
          #echo "Hyper: $strategy: $result"
          hyperResults[${strategy}]="${result}"
          state=$st_find_params
        fi

      elif [ $state -eq $st_find_params ]; then
        # scan until "buy_params" found
        if [[ $line =~ .*"buy_params".* ]]; then
          strat_params=""
          count=0
          state=$st_params
        fi

      elif [ $state -eq $st_params ]; then
        if [[ $line =~ .*"}".* ]]; then
          hyperParams[${strategy}]="${strat_params}"
          #echo "strat_params: " "${strat_params}"
          state=$st_find_name
        else
          line="${line// /!}" # replace spaces
          if [ $count -gt 0 ]; then
            strat_params="$strat_params@${line}" # no multi-dimensional arrays in bash, form string instead
          else
            strat_params="${line}"
          fi
          count=$((count + 1))

        fi

      else
        echo "ERROR: invalid state: ${state}"
      fi
    done < "${hyperfile}"
    #echo "#=========================================="
  
  else
    echo "ERROR: ${hyperfile} not found"
  fi
  
  set +x
}

# func to compare test results to hyperopt results
compare_results () {

  params=""

  echo ""
  echo "Exchange: ${exchange}"
  echo ""
  printf "%10s %10s %10s %10s\n" "Strategy" "Test" "Hyperopt" "Opinion"
  printf "%10s %10s %10s %10s\n" "--------" "----" "--------" "-------"

  RED='\033[0;31m'
  GREEN='\033[0;32m'
  NC='\033[0m' # No Color
  #for s in "${!testResults[@]}"; do
  for s in "${strategies[@]}"; do

    if [[ -v hyperResults[$s] ]]; then

      #echo "$s"
      tr=${testResults["$s"]}
      hr=${hyperResults["$s"]}
      if awk "BEGIN {exit !($hr == $tr)}"; then
        printf "%-10s %10s %10s %10s\n" "$s" "$tr" "$hr" "Leave"
      elif awk "BEGIN {exit !($hr >= $tr)}"; then
        printf "%-10s %10s %10s ${GREEN}%10s${NC}\n" "$s" "$tr" "$hr" "Update"
      else
        printf "%-10s %10s ${RED}%10s${NC} ${RED}%10s${NC}\n" "$s" "$tr" "$hr" "Leave"
      fi
    else
      echo "ERR: strategy $s not found in Hyperopt results"
    fi
  done

  # print the parameters
  echo ""
  echo "#HyperParameters"
  echo ""
  for s in "${strategies[@]}"; do

    if [[ -v hyperResults[$s] ]]; then

      #echo "$s"
      tr=${testResults["$s"]}
      hr=${hyperResults["$s"]}

      if awk "BEGIN {exit !($hr > $tr)}"; then
        params=()
        str="${hyperParams[${s}]}"
        params=(${str//@/ }) # split string

        echo "# $s"
        echo "strategyParameters[\"$s\"] = {"

        len=${#params[@]}
        #echo "len: $len"
        for (( i=0; i<${len}; i++ )); do
          params[$i]="${params[$i]//!/ }" # restore spaces
          echo "${params[$i]}"
        done
        echo "}"
        echo ""
      fi
    else
      echo "ERR: strategy $s not found in Hyperopt results"
    fi
  done

}

# Main Code

if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    exit 0
fi

exchange=$1
testfile="test_${exchange}.log"
hyperfile="hyp_${exchange}.log"

# check that files exist
if [ ! -f ${testfile} ]; then
    echo "test log file not found: ${testfile}"
    exit 0
fi

if [ ! -f ${hyperfile} ]; then
    echo "hyperopt log file not found: ${hyperfile}"
    exit 0
fi

scan_test_file
scan_hyper_file
compare_results