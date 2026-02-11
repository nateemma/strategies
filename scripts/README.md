# Helper Scripts

The majority of scripts here are to help deal with multiple exchanges, or groups of strategies.

NOTE: I use Macs for development, so most of these scripts assume zsh (the default shell on Macs). 
You may run into issues if you try to use bash instead. Most of teh scripts print out the freqtrade command that they are using, so you can just copy and modify that if needed.

| Script                          | Description                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| cleanup.sh                      | Removes 'old' files from user_data subdirectories (hyperopt, backtesting, plots etc.). Default is to remove anything older than 30 days. |
| download.sh                     | Downloads candle data for an exchange. Defaults to all exchanges                                                                         |
| test_strat.sh                   | Tests an individual strategy for the specified exchange                                                                                  |
| hyp_strat.sh                    | runs hyperopt on an individual strategy for the specified exchange                                                                       |
| dryrun_strat.sh                 | Dry-runs a strategy on the specified exchange, takes care of PYTHONPATH, db-url etc                                                      |
| run_strat.sh                    | Runs a strategy live on the specified exchange, takes care of PYTHONPATH, db-url etc                                                     |
| test_group.sh                   | Tests a group of strategies and summarises the results. Useful because it takes wildcards                                                |
| hyp_group.sh                    | Runs hyperopt on a group of strategies (with wildcards)                                                                                  |
| SummariseTestResults.py         | Summarises the output of test_group.sh (or any backtest file). Note: python, not shell script                                            |
| SummariseHyperOptTestResults.py | Summarises the output of hyp_group.sh (or any hyperopt output)                                                                           |
| ShowTestResults.py              | The Summarise*.py scripts save the results to a json file. This script displays those results as a table                                 |

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

## Short Trading
Shorting requires the use of a separate config file and (curently) static pairlists. 
The scripts have been updated to accept a _--short_ argument that will then use the config file 
config\_<exchange>_short.json, which must be in the exchange directory.

Note that download and hyperopt for short-based strategies is *much* slower
