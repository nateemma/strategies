# Phil's Custom freqtrade Crypto Trading Strategies

This folder contains the code for a variety of custom trading strategies for use with the [freqtrade](https://www.freqtrade.io/) framework.


Note: as of November 2021, I abondoned work on the "Combo" strategies, which have been moved to the _archived/_ folder<br>
Instead, I now focus on strategies derived from the FisherBB strategy, which uses a very simple buy signal based on the 
Fisher indicator and Bollinger Bands.



## Disclaimer

These strategies are for educational purposes only


Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME 
NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by testing strategies using backtesting then run the trading bot in Dry-run. 
Do not engage money before you understand how it works and what profit/loss you should expect. 
Also, do not backtest the strategies in a period of huge growth (like 2020 for example), 
when any strategy would do well. I recommend including May 2021, which includes a pretty big crash (-50%).

## Reference repositories
I either used or learned from strategies in the github repositories below:

https://github.com/freqtrade/freqtrade-strategies
<br>
https://github.com/i1ya/freqtrade-strategies
<br>
https://github.com/ntsd/freqtrade-configs
<br>
https://github.com/froggleston/cryptofrog-strategies
<br>
https://github.com/werkkrew/freqtrade-strategies
<br>
https://github.com/brookmiles/freqtrade-stuff
<br>
https://github.com/hansen1015/freqtrade_strategy/blob/main/heikin.py
<br>
https://github.com/Foxel05/freqtrade-stuff


## Multi Exchange Support
Exchanges have different pairlists, different base currencies and different behaviours, so you really need different 
parameters for each exchange. The way I address this is by putting exchange-specific code in a subdirectory below the 
strategy directory whose name matches the exchange tag used by freqtrade (e.g. binanceus, kucoin, ftx). 
<br>
If you look, you will find files such as FisherBBWtdProfit.py in each subdirectory. These files contain exchange-specific code, 
which is usually the set of hyperparameters for the strategies, plus the ROI, stoploss and 
trailing parameters.

Because the FisherBB base class is in the user_data/strategies directory, you have to manage the _PYTHON_PATH_ environment 
variable or import statements will not find the correct files. 
<br>
To help with this, I added a bunch of shell scripts in the _user_data/strategies/scripts_ directory:

| Script | Description |
|-----------|------------------------------------------|
|test_strat.sh|Tests an individual strategy for the specified exchange |
|test_exchange.sh|Tests all of the currently active strategies for the specified exchange |
|test_monthly.sh| Runs test_exchange.sh over a monthly interval for the past 6 months, shows average performance, and ranks the strategies |
|hyp_strat.sh|runs hyperopt on an individual strategy for the specified exchange |
|hyp_exchange.sh|Runs hyp_strat.sh for all of the currently active strategies for the specified exchange |
|hyp_all.sh| Runs hyp_exchange.sh for all currently active exchanges (takes a *_very_* long time) |
|run_strat.sh| Runs a strategy live on the specified exchange, takes care of PYTHONPATH, db-url etc |
|dryrun_strat.sh| Dry-runs a strategy on the specified exchange, takes care of PYTHONPATH, db-url etc |


Specify the -h option for help.

Please note that all of the _test__\*.sh and _hyp__\*.sh scripts all expect there to be a config file in the exchange directory that is named in the form:  
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


Also, 


## Setting Up Your Configuration

See the [freqtrade docs](https://www.freqtrade.io/en/stable/configuration/) for generic instructions. 

My environment is set up for multiple exchanges, so it's a bit different. My scripts expect the following:

* an exchange-specific config file in the root _freqtrade_ directory of the form config\_\<exchange\>.json 
(or config_\<exchange\>_\<port\>.json if you want to run multiple strategies on the same exchange)
* an exchange-specific config file in the user_data/strategies/\<exchange\> directory of the form config\_\<exchange\>.json

Note: do *_not_* put any exchange trading keys or passwords in the user_data/strategies/\<exchange\> files, as you are quite likely to share these or put them into github

## Downloading Test Data
To run backtest and hyperopt, you need to download data to your local environment.
<br>
ou do this using a command like this:

>freqtrade download-data  --timerange=_\<timerange\>_ -c _\<config\>_-t 5m 15m 1h 1d

Or, for a specific pair (e.g. BTC/USDT):

>freqtrade download-data  --timerange=_\<timerange\>_ -c _\<config\>_-t 5m 15m 1h 1d -p BTC/USDT

Most of the FisherBB strategies only need 15m data, but some of the more advanced sell logic also requires 15m, 1h and 1d data, 
plus also data for BTC/USD or BTC/USDT (whatever your exchanges uses)

I typicaly download for the timerange 20210501- (May 1st, 2021 to present)


## Backtesting

Backtesting is where you run your strategy on historical data to see how it would gave performed.

In all command/shell examples, _\<strategy\>_ is the name of the strategy being used, and _\<timerange\>_ is the range of time to use for testing (duh).

Examples of _\<timerange\>_ might be:
- _20210501-_       May 1st, 2021 until present date
- _20210603-20210605_   June 3rd 2021 until June 5th 2021

Before you can backtest, you must download data for testing. 

It is recommended that you update fairly often (e.g. once per week) and on a limited time range (e.g. the last month). 
Optimising results for the past year or 6 months doesn't really help you perform better with the current market conditions.

Backtesting can be done using the following command:

>freqtrade backtesting  -c _\<config\>_ --strategy _\<strategy\>_  --timerange=_\<timerange\>_

## Hyper-Parameter Optimisation

Run the hyperopt command to search for 'optimal' parameters in your strategy. 

Note that performance is _dramatically_ affected by the ROI parameters and timeframe. 

>freqtrade hyperopt  -c _\<config\>_ --strategy _\<strategy\>_  --space _\<space\>_ --hyperopt-loss _\<loss algorithm\>_   --timerange=_\<timerange\>_

where:

_\<space\>_ specifies which space to try and optimise. It is recommended to try these combinations in order:

1. buy (using SharpeHyperOptLossDaily)
2. roi stoploss  (using OnlyProfitHyperOptLoss)
3. trailing  (using OnlyProfitHyperOptLoss)


If you try to optimise them all at once, you don't get good results. Also, note the use of different loss functions above


In my case, I tend to just run all of these spaces only on ComboHold, since that is what I actually use to trade. 
I also run the _buy_ space on all of the strategies that go into ComboHold, since it changes a lot week-to-week.



If the run gets better results than before, update your strategy with the suggested parameters (roi table etc). Note that _hyperopt-loss_ does not always provide better results

_\<loss algorithm\>_ is one of:

- ShortTradeDurHyperOptLoss
- OnlyProfitHyperOptLoss
- SharpeHyperOptLoss
- SharpeHyperOptLossDaily
- SortinoHyperOptLoss
- SortinoHyperOptLossDaily

Run them in that order and see if you get different results

## Custom Hyperopt Loss Functions

## Plotting Results

It is often very useful to see a visual representation of your strategy. You can do this using the plot-dataframe command:

>freqtrade plot-dataframe --strategy _\<strategy\>_  -p BCH/USD --timerange=_\<timerange\>_ --indicators1 ema5 ema20 --indicators2 mfi

This example creates a plot for pair BCH/USD, adds ema5 and ema20 to the main chart and adds a plot of mfi below that (these must exist in your dataframe for the strategy). You can find it in the _user_data/plot/_ directory, and in this case the name of the file would be: 
_user_data/plot/freqtrade-plot-BCH_USD-5m.html_
<p>You can open this in a browser, and you can zoom in/out and move around</p>
Any buys and sells generated by backtesting (if any) will be shown on the plot, which is very useful for debugging.

Tip: constrain the plot to a day or 2, otherwise you have to zoom in a lot. Also, when debugging, use the -p option in backtesting, which will cause the strategy to run only on the pair that you specify, and not across the whole list of pairs.
Also, choose the pair that has the worst performance in your strategy to see where things are going wrong.

## Dry Runs

Easy. In a command window, just run:

>freqtrade trade --dry-run --strategy _\<strategy\>_

If you install freqUI (instructions [here](https://www.freqtrade.io/en/stable/rest-api/)), then you can 
monitor the trades on a web page at http://127.0.0.1:8080/ (or whatever address you specify in config.json)

## Live Trading

For this, you need an account on an exchange, and you need to modify config.json to specify which exchange, trading limits, private keys etc.
You need to get API keys from the exchange website, but that is usually easy.

In a command window, just run:

>freqtrade trade --strategy _\<strategy\>_

You can monitor trades from the UI (see above), and from the exchange website/app

Note that you need your computer synched up to an NTP time source to do this.

## List of Strategies

The following is a list of (most of) my custom strategies and a rough assessment of performance. 
They are listed in order of relative performance, good to bad, but please note that performance depends a lot upon the 
time period being tested, which exchange you are using, and which pairs you are trading.

<br>
Most of the strategies are contained in exchange-specific folders (binance/ etc.) because the hyperparameters are very 
specific for each exchnage.

Note: I am pursuing a "buy on dips, sell at a profit" approach, so most of these strategies
do not issue sell signals at all. Instead, they use the ROI and stoploss mechanisms of freqtrade to sell (hopefully at a profit).
I started off trying to issue sell signals, but performance was bad and I've had far better results just holding.<br>
FYI, this makes the strategies *very* dependent upon the exchange and pairlist 

Most of the different strategies are actually just the FisherBB strategy 'tuned' for a specific exzchange, and using a 
different hyperopt loss function (See the section [Custom Hyperopt Loss Functions](#custom-hyperopt-loss-functions))


|        Strategy |    Description                                 | 
|-----------------|------------------------------------------------|
| FisherBB | Base class that implements the buy logic for the Fisher and Bollinger Band indicators<br>Other _FisherBB*_ strategies usually derive from this and just change parameters |
| FisherBBWtdProfit | FisherBB tuned using the WeightedProfitHyperOptLoss function |
| FisherBBQuick |  FisherBB tuned using the QuickProfitHyperOptLoss function |
| FisherBBExp |  FisherBB tuned using the ExpectancyHyperOptLoss function |
| FisherBBPED |  FisherBB tuned using the PEDHyperOptLoss function |
| FisherBBWinLoss |  FisherBB tuned using the WinHyperOptLoss function |
| FisherBBDynamic | A merge of FisherBB (buy) and the dynamic ROI logic adapted from MacheteV8b (see [Foxel05](https://github.com/Foxel05/freqtrade-stuff) repo)|
| FisherBBSolipsis | A merge of FisherBB (buy) and the custom sell logic adapted from Solipsis_V5 (see [werkkrew](https://github.com/werkkrew/freqtrade-strategies) repo)|
