# Helper Scripts

The majority of scripts here are to help deal with multiple exchanges.

NOTE: I use Macs for development, so most of these scripts assume zsh (the default shell on Macs). 
You may run into issues if you try to use bash instead.

| Script | Description |
|-----------|------------------------------------------|
|cleanup.sh| Removes 'old' files from user_data subdirectories (hyperopt, backtesting, plots etc.). Default is to remove anything older than 30 days.|
|compareStats.sh|Parses output from test_monthly.sh and summarises results across suppoirted exchanges|
|download.sh|Downloads candle data for an exchange. Defaults to all exchanges|
|dryrun_strat.sh| Dry-runs a strategy on the specified exchange, takes care of PYTHONPATH, db-url etc |
|hyp_strat.sh|runs hyperopt on an individual strategy for the specified exchange |
|hyp_exchange.sh|Runs hyp_strat.sh for all of the currently active strategies for the specified exchange |
|hyp_all.sh| Runs hyp_exchange.sh for all currently active exchanges (takes a *_very_* long time) |
|run_strat.sh| Runs a strategy live on the specified exchange, takes care of PYTHONPATH, db-url etc |
|test_strat.sh|Tests an individual strategy for the specified exchange |
|test_exchange.sh|Tests all of the currently active strategies for the specified exchange |
|test_monthly.sh| Runs test_exchange.sh over a monthly interval for the past 6 months, shows average performance, and ranks the strategies |


Specify the -h option for help.

Please note that most of the scripts all expect there to be a config file in the exchange directory that is named in the form:  
_config\_<exchange>.json_ (e.g. _config_binanceus.json_)
<br>
The _run_strat.sh_.sh and _dryrun_strat.sh_ scripts expect a 'real' config file that should specify volume filters etc.

I include reference config files for each exchange (in each exchange folder). These use static pairlists since VolumePairlist does not work for backtesting or hyperopt. 
<br>To generate a candidate pairlist use a command such as the following:

>freqtrade test-pairlist -c _\<config\>_

where _\<config\>_ is the 'real' config file that would be used for live or dry runs.
<br>
the command will give a list of pairs that pass the filters. You can then cut&paste the list into your test config file. 
Remember to change single quotes (\') to double quotes (\") though.

I find that I need to refresh the list every few weeks.

