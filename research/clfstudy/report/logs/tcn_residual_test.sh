#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/RNN_SWEEP_DONE ]; do sleep 120; done
for s in 0 1 2; do
python -m research.clfstudy.run --config '{"arch":"TCN_FAM","hp":{"channels":64,"depth":8,"dilation_base":2,"residual":true},"input_set":"ind24","seq_len":16,"split_set":"v1","phase":0,"train":{"lr":0.0003,"stride":2},"tags":["tcn-ablation","depth8_dil2_residual"]}' --window C --seed $s
done
echo DONE > research/clfstudy/report/logs/TCN_RESIDUAL_DONE
