#!/bin/zsh

# just a script to run stuff overnight

echo ""
echo ""

#zsh user_data/strategies/scripts/download.sh binanceus

pca_list=("PCA_dwt" "PCA_fbb" "PCA_highlow" "PCA_jump" "PCA_macd" "PCA_mfi" "PCA_minmax" "PCA_nseq" "PCA_over" \
"PCA_profit" "PCA_stochastic" "PCA_swing")

nnbc_list=("NNBC_fbb" "NNBC_jump" "NNBC_mfi" "NNBC_minmax" "NNBC_nseq" "NNBC_profit" "NNBC_swing")

nnp_list=("NNPredict" "NNPredict_Multihead" "NNPredict_Transformer" "NNPredict_MLP")

logfile="overnight.log"
today=`date`
echo "${today} overnight.sh" >$logfile

for strat in $nnp_list; do
  echo "" >>$logfile
  echo "-------------------" >>$logfile
  echo "${strat}" >>$logfile
  echo "-------------------" >>$logfile
  echo "" >>$logfile
  zsh user_data/strategies/scripts/test_strat.sh -n 750 binanceus ${strat} >>$logfile
done

python user_data/strategies/scripts/SummariseTestResults.py $logfile
