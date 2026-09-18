#!/bin/bash
# Fair conv-family test: TCN_FAM with Wavenet's exact schedule (channels 32, kernel 2, dilations [1,2,4,8]x2), window C, 3 seeds.
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
until [ -f research/clfstudy/report/logs/RNN_SWEEP_DONE ]; do sleep 120; done
for s in 0 1 2; do
python -m research.clfstudy.run --config '{"arch":"TCN_FAM","hp":{"channels":32,"kernel":2,"dilations":[1,2,4,8,1,2,4,8]},"input_set":"ind24","seq_len":16,"split_set":"v1","phase":0,"train":{"lr":0.0003,"stride":2},"tags":["tcn-ablation","wavenet_clone"]}' --window C --seed $s
done
echo DONE > research/clfstudy/report/logs/TCN_CLONE_DONE
