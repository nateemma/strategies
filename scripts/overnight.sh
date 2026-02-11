#!/bin/zsh

# just a script to run stuff overnight

echo ""
echo ""

#zsh user_data/strategies/scripts/download.sh binanceus

strat_type="PCA"

run_hyp=true

list=()

if [[ $# -gt 0 ]]; then
  strat_type="${1}"
fi



#list=$(exec find user_data/strategies/binanceus/${strat_type}_*.py -type f -exec basename {}  -print0 \;)

list=()
dir="./user_data/strategies/binanceus"
files="${strat_type}_*.py"
find_result=$( find ${dir} -name "${files}" -type f -print0 | xargs -0 basename )

#echo "find_result: " ${find_result}

list=( "${(@f)${find_result}}" )

num_files=${#list[@]}
if [[ $num_files -eq 0 ]]; then
  echo "ERR: no files found"
  return
fi

#echo "Files: ${list}"

typeset -l logfile # force lowercase
typeset -l hyplog

logfile="overnight_${strat_type}.log"
hyplog="overnight_hyp_${strat_type}.log"

echo "Strategy list: ${list}"
echo "Test log:      ${logfile}"

today=$(date)
echo "" >$logfile
echo "============================" >>$logfile
echo "${today} overnight.sh" >>$logfile
echo "============================" >>$logfile
echo "" >>$logfile

echo "\${run_hyp}:  ${run_hyp}"

if $run_hyp ; then
  echo "HyperOpt log:  ${hyplog}"
  echo "" >$hyplog
  echo "============================" >>$hyplog
  echo "${today} overnight.sh" >>$hyplog
  echo "============================" >>$hyplog
  echo "" >>$hyplog
fi

for file in ${list}; do
  strat=$file:t:r
  echo $strat

  echo ""
  echo "-------------------"
  echo "${strat}"
  echo "-------------------"
  echo ""

  if ${run_hyp} ; then
    echo "" >>$hyplog
    echo "-------------------" >>$hyplog
    echo "${strat}" >>$hyplog
    echo "-------------------" >>$hyplog
    echo "" >>$hyplog
    zsh user_data/strategies/scripts/hyp_strat.sh -n 90 -e 100 -s sell -l CalmarHyperOptLoss binanceus ${strat} >>$hyplog
  fi

  echo "" >>$logfile
  echo "-------------------" >>$logfile
  echo "${strat}" >>$logfile
  echo "-------------------" >>$logfile
  echo "" >>$logfile
  zsh user_data/strategies/scripts/test_strat.sh -n 30 binanceus ${strat} >>$logfile

done
echo "============================" >>$logfile

python user_data/strategies/scripts/SummariseTestResults.py ${logfile}

if $run_hyp ; then
  python user_data/strategies/scripts/SummariseHyperOptResults.py ${hyplog}
fi

#cat $logfile

echo ""
echo "Output log:${logfile}"
echo ""