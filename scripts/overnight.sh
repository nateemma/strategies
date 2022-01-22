#!/bin/zsh

# test strategies, run buy parameter hyperopt and compare results for each exchange
# Run overnight, this will take many hours!

echo ""
echo ""

zsh user_data/strategies/scripts/download.sh

zsh user_data/strategies/scripts/hyp_all.sh

zsh user_data/strategies/scripts/updateMonthly.sh