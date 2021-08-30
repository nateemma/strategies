#! /bin/bash

# script to scan hyperall.log and generate the corresponding python that can be used to update ComboHoldParams.py
# output is to stdout


if [[ $# -eq 0 ]] ; then
    echo 'please specify exchange'
    return 0
fi

exchange=$1
logfile="hyperall_${exchange}.log"

if [ ! -f ${logfile} ]; then
    echo "log file not found: ${logfile}"
    return 0
fi


# states
st_find_name=0
st_name=1
st_find_params=2
st_params=3
state=$st_find_name
strategy=""

#set -x
set +x

# check file
if [ -f "${logfile}" ]; then
  echo ""
  echo "#====== AUTO-GENERATED PARAMETERS ========="
  echo ""
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
      echo "# ${strategy}"
      state=$st_find_params

    elif [ $state -eq $st_find_params ]; then
      # scan until buy_params found
      if [[ $line =~ .*"buy_params = {".* ]]; then
        echo "strategyParameters[\"${strategy}\"] = {"
        state=$st_params
      fi
      if [[ $line =~ .*"No epochs evaluated yet".* ]]; then
        echo "# No parameters found!"
        echo ""
        state=$st_find_name
      fi

    elif [ $state -eq $st_params ]; then
      if [[ $line =~ .*"}".* ]]; then
        # end of params
        echo "}"
        echo ""
        state=$st_find_name
      else
        # copy line (and trim leading whitespace)
        echo "    ${line#"${line%%[![:space:]]*}"}"
      fi
    else
      echo "ERROR: invalid state: ${state}"
    fi
  done < "${logfile}"
  echo "#=========================================="

else
  echo "ERROR: ${logfile} not found"
fi

set +x