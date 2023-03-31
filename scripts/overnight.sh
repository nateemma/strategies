#!/bin/zsh

# just a script to run stuff overnight

echo ""
echo ""

#zsh user_data/strategies/scripts/download.sh binanceus

strat_type="PCA"

list=()

if [[ $# -gt 0 ]]; then
  strat_type="${1}"
fi

#if [[ "${strat_type}" == "pca" ]]; then
#  list=("PCA_dwt" "PCA_fbb" "PCA_highlow" "PCA_jump" "PCA_macd" "PCA_mfi" "PCA_minmax" "PCA_nseq" "PCA_over" \
#"PCA_profit" "PCA_stochastic" "PCA_swing")
#elif  [[ "${strat_type}" == "anomaly" ]]; then
#  list=("Anomaly_dwt" "Anomaly_macd" "Anomaly_nseq" "Anomaly_profit" )
#elif  [[ "${strat_type}" == "nnbc" ]]; then
#  list=("NNBC_fbb" "NNBC_jump" "NNBC_minmax" "NNBC_nseq" "NNBC_profit" "NNBC_swing")
#elif  [[ "${strat_type}" == "nnpredict" ]]; then
#  list=("NNPredict" "NNPredict_Multihead" "NNPredict_Transformer" "NNPredict_MLP")
#elif  [[ "${strat_type}" == "nntc" ]]; then
#  list=("NNTC_highlow_LSTM" "NNTC_profit_GRU"  \
#  "NNTC_dwt_LSTM" "NNTC_macd_GRU"  "NNTC_profit_LSTM" \
#  "NNTC_macd_LSTM" "NNTC_profit_LSTM2" \
#  "NNTC_fbb_GRU" "NNTC_macd_Multihead" "NNTC_profit_Wavenet" \
#  "NNTC_fbb_LSTM" "NNTC_nseq_GRU" "NNTC_pv_LSTM" \
#  "NNTC_fbb_Multihead" "NNTC_nseq_LSTM" "NNTC_pv_Multihead" \
#  "NNTC_fbb_Wavenet" "NNTC_nseq_Transformer" "NNTC_pv_Wavenet" \
#  "NNTC_fwr_LSTM" "NNTC_nseq_Wavenet" "NNTC_swing_LSTM")
#
#else
#  echo "ERR: unknown strategy list: ${1}"
#  return
#fi


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

logfile="overnight_${strat_type}.log"

echo "Strategy list: ${list}"
echo "Output log:    ${logfile}"

today=$(date)
echo "" >$logfile
echo "============================" >>$logfile
echo "${today} overnight.sh" >>$logfile
echo "============================" >>$logfile
echo "" >>$logfile

for file in ${list}; do
  strat=$file:t:r
  echo $strat
  echo "" >>$logfile
  echo "-------------------" >>$logfile
  echo "${strat}" >>$logfile
  echo "-------------------" >>$logfile
  echo "" >>$logfile
#  zsh user_data/strategies/scripts/test_strat.sh -n 750 binanceus ${strat} >>$logfile
  zsh user_data/strategies/scripts/test_strat.sh -n 60 binanceus ${strat} >>$logfile
#  zsh user_data/strategies/scripts/hyp_strat.sh -n 90 -e 100 -s sell -l CalmarHyperOptLoss binanceus ${strat} >>$logfile
done
echo "============================" >>$logfile

python user_data/strategies/scripts/SummariseTestResults.py ${logfile}

#cat $logfile

echo ""
echo "Output log:${logfile}"
echo ""