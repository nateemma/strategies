#!/bin/zsh

# just a script to run stuff overnight

echo ""
echo ""

#zsh user_data/strategies/scripts/download.sh binanceus


nnbc_list=("NNBC_fbb" "NNBC_jump" "NNBC_mfi" "NNBC_minmax" "NNBC_nseq" "NNBC_profit" "NNBC_swing")

nnp_list=("NNPredict" "NNPredict_Attention" "NNPredict_Multihead" "NNPredict_Transformer" "NNPredict_LSTM2" \
"NNPredict_MLP" "NNPredict_MLP2")

for strat in nnp_list; do
  zsh user_data/strategies/scripts/test_strat.sh -n 365 binanceus ${strat}
done
