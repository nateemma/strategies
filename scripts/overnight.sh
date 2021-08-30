#!/bin/bash

# test strategies, run buy parameter hyperopt and compare results for each exchange
# Run overnight, this will take many hours!

echo "=================="
echo "Testing Strategies"
echo "=================="
bash user_data/strategies/scripts/testall.sh binanceus
bash user_data/strategies/scripts/testall.sh kucoin
bash user_data/strategies/scripts/testall.sh coinbasepro

echo "=============================="
echo "Running Hyperopt on Strategies"
echo "=============================="
bash user_data/strategies/scripts/hyperall.sh binanceus
bash user_data/strategies/scripts/hyperall.sh kucoin
bash user_data/strategies/scripts/hyperall.sh coinbasepro

echo "=========================================="
echo "Comparing Test Results to Hyperopt Results"
echo "=========================================="
bash user_data/strategies/scripts/compareResults binanceus
bash user_data/strategies/scripts/compareResults kucoin
bash user_data/strategies/scripts/compareResults coinbasepro
