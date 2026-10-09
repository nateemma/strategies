#!/bin/bash
cd /Users/philprice95/projects/freqtrade/user_data/strategies
source /Users/philprice95/projects/freqtrade/.venv/bin/activate; export PYTHONPATH=.
for s in 0 1 2; do
python -m research.clfstudy.run --config '{"arch":"LSTM","input_set":"ind24","seq_len":16,"split_set":"v1","phase":0,"train":{"lr":0.0003,"stride":2},"tags":["protocol-calibration","lr3e-4-stride2"]}' --window C --seed $s
done
echo DONE > research/clfstudy/report/logs/PROTOCOL_CALIB2_DONE
