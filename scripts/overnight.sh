#!/bin/zsh

# just a script to run stuff overnight

echo ""
echo ""

#zsh user_data/strategies/scripts/download.sh binanceus

strat_type="pca"

list=()

if [[ $# -gt 0 ]]; then
  strat_type="${1}"
fi

if [[ "${strat_type}" == "pca" ]]; then
  list=("PCA_dwt" "PCA_fbb" "PCA_highlow" "PCA_jump" "PCA_macd" "PCA_mfi" "PCA_minmax" "PCA_nseq" "PCA_over" \
"PCA_profit" "PCA_stochastic" "PCA_swing")
elif  [[ "${strat_type}" == "anomaly" ]]; then
  list=("Anomaly_dwt" "Anomaly_macd" "Anomaly_minmax" "Anomaly_profit" )
elif  [[ "${strat_type}" == "nnbc" ]]; then
  list=("NNBC_fbb" "NNBC_jump" "NNBC_minmax" "NNBC_nseq" "NNBC_profit" "NNBC_swing")
elif  [[ "${strat_type}" == "nnpredict" ]]; then
  list=("NNPredict" "NNPredict_Multihead" "NNPredict_Transformer" "NNPredict_MLP")
else
  echo "ERR: unknown strategy list: ${1}"
  return
fi

logfile="overnight_${strat_type}.log"

echo "Strategy list: ${list}"
echo "Output log:    ${logfile}"

today=$(date)
echo "${today} overnight.sh" >"$logfile"

for strat in $list; do
  echo "" >>"$logfile"
  echo "-------------------" >>"$logfile"
  echo "${strat}" >>"$logfile"
  echo "-------------------" >>"$logfile"
  echo "" >>"$logfile"
  zsh user_data/strategies/scripts/test_strat.sh -n 750 binanceus ${strat} >>"$logfile"
done

python user_data/strategies/scripts/SummariseTestResults.py $logfile

cat "$logfile"

echo ""
echo "Output log:${logfile}"
echo ""

