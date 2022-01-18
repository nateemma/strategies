# Phil's Custom freqtrade Crypto Trading Strategies

This folder contains the code for a variety of custom trading strategies for use with the [freqtrade](https://www.freqtrade.io/) framework.

Please read through the instructions at https://www.freqtrade.io before attempting to use this software.

Note: I have tried many different strategies, most of which perform quite badly or inconsistently. 
The abandoned strategies are in the _archived/_ folder<br> for reference (I sometimes cut & paste pieces of them into 
new strategies).

Currently, I now focus on strategies derived from a single buy strategy, which uses a very simple buy signal based on the 
Fisher RSI indicator, Bollinger Bands and Williams %R indicator. I typically combine the base buy signals with various 
guards, triggers and sell approaches. Where I use code from someone else's strategy, I will reference that in the 
comments, and there is a list of useful strategy repositories below if you need more ideas. I tend to use a naming 
convention of "FBB_" followed by the sell strategy. For example, FBB_ROI indicates Fisher/Bollinger Band buy signals 
with an ROI sell signal (meaning just using the ROI mechanism to determine sell points). Similarly FBB_Solipsis uses (some of) the
sell approach from the Solipsis strategy (by werkrew).



## Disclaimer

These strategies are for educational purposes only


Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME 
NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by testing strategies using backtesting then run the trading bot in Dry-run mode (live data but simulated trades). 
Some instructions on how to do this are provided below.
<br>Never, _ever_, go to live trading without first going through a dry-run - it is not at all uncommon for a strategy to 
achieve fantastic results in backtetsting, only to perform very badly in a live situation. 
The main reason for this is that the backtesting simulation cannot reproduce the behaviour of the live environment. 
For example, real trades take a relatively long time and the price can move significantly during that time. Also, market 
conditions such as trading volume, spreads, volatility etc. cannot be reproduced from the historical data provided by the exchanges

Do not engage money before you understand how it works and what profit/loss you should expect. 
Also, do not backtest the strategies in a period of huge growth (like 2020 for example), 
when any strategy would do well. I recommend including periods where the market performed poorly (e.g. May, Nov and Dec 2021)


## List of Strategies

The following is a list of my custom strategies that I am currently testing. 

| Strategy         | Description                                                                                                                                                                               | 
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FBB_ROI          | buy logic using the Fisher and Bollinger Band indicators (and recently Williams %R). <br>Most other strategies here use the same buy signals                                              |                                                                        |
| FBB_2            | FBB buy signals, combined with some correlation to BTC, and sell signals based on Fisher RSI and Bollinger Bands.<br>(the "2" is meant indicate 'squared", i.e. FBB on both buy and sell. | |
| FBB_2Sqz         | FBB_2 with addition of SqueezeMomentum indicators.<br>Based on TradingView strategy by LazyBear                                                                                           |
| FBB_Leveraged    | FBB applied to leveraged pairs (e.g. ADABULL, ADABEAR).                                                                                                                                   |
| FBB_BTCLeveraged | Same as FBB_Leveraged, but uses signals based on BTC up/down trends                                                                                                                       |
| FBB_Solipsis     | A merge of FBB_ (buy) and the custom sell logic adapted from Solipsis_V5 (see [werkkrew](https://github.com/werkkrew/freqtrade-strategies) repo)                                          |

Please note that you will need both the _.py_ *and* the _.json_ file.

If you know what you are doing, go ahead and use these (but read the section on muliple exchanges first). 
If not, please read through the gneral freqtrade documentation and the guidelines below...

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
If you look, you will find files such as FBB_ROI.py in each subdirectory. 
The python files are usually identical across the exchange directories, but the associated json file 
(e.g. FBB_ROI.json) contains the exchange-specific data for the strategy, which is typically generated by the 
_freqtrade hyperopt_ command (or the hyp_strat.sh script)

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
You do this using a command like this:

>freqtrade download-data  --timerange=_\<timerange\>_ -c _\<config\>_-t 5m 15m 1h 1d

Or, for a specific pair (e.g. BTC/USDT):

>freqtrade download-data  --timerange=_\<timerange\>_ -c _\<config\>_-t 5m 15m 1h 1d -p BTC/USDT

Most of the FBB_ strategies only need 5m data, but some of the more advanced sell logic also requires 15m, 1h and 1d data, 
plus also data for BTC/USD or BTC/USDT (whatever your exchanges uses)

I typically download for the past 180 days, which is the default used by the various scripts.

For convenience, you can also just run :

>bash user_data/strategies/scripts/download.sh


## Backtesting

Backtesting is where you run your strategy on historical data to see how it would have performed.

In all command/shell examples, _\<strategy\>_ is the name of the strategy being used, and _\<timerange\>_ is the range of time to use for testing (duh).

Examples of _\<timerange\>_ might be:
- _20210501-_       May 1st, 2021 until present date
- _20210603-20210605_   June 3rd 2021 until June 5th 2021

Before you can backtest, you must download data for testing. 

It is recommended that you update fairly often (e.g. once per week) and on a limited time range (e.g. the last month). 
Optimising results for the past year or 6 months doesn't really help you perform better with the current market conditions.

Backtesting can be done using the following command:

>freqtrade backtesting  -c _\<config\>_ --strategy-path _\<path\>_ --strategy _\<strategy\>_  --timerange=_\<timerange\>_

Or, you can use a script:

>bash user_data/strategies/scripts/test_strat.sh _\<exchange\>_ _\<strategy\>_

Use the -h option for options. 

## Hyper-Parameter Optimisation

Run the hyperopt command to search for 'optimal' parameters in your strategy. 

The parameters that can be tuned are defined in the strategy using calls such as:

>   buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)

In this example, the code is telling freqtrade that the strategy has a _buy_ parameter named _buy\_bb\_gain_, which is 
a floating point number (Decimal) between 0.01 and 0.1, using 2 decimal places

Note that performance is _dramatically_ affected by the parameters (especially ROI) and timeframe. 
Also, strategies tuned for a specific exchange will typically not perform well on other exchanges.

The freqtrade command to run the optimisation is:

>freqtrade hyperopt  -c _\<config\>_ --strategy-path _\<path\>_ --strategy _\<strategy\>_  --spaces _\<space\>_ --hyperopt-loss _\<loss algorithm\>_   --timerange=_\<timerange\>_

where:

_\<space\>_ specifies which space to try and optimise. Typical options are buy, sell, roi, stoploss and trailing. 
I tend to _not_ optimise for stoploss, and just set it manually to 10% (-0.1), or 99% for strategies that use 
dynamic ROI or a custom stoploss (e.g. FBB_Solipsis)

Or, you can use a script:

>bash user_data/strategies/scripts/hyp_strat.sh -s "buy sell roi" _\<exchange\>_ _\<strategy\>_

If you have a fast computer, try _--space buy sell roi_ and then _--spaces trailing_, but if you have a slower computer, try them in sequence.

NOTE: the optimised parameters are written to a json file that matches the strategy file, e.g. FBB_ROI.py will produce FBB_ROI.json.
freqtrade commands (backtesting, dryrun, live running) will take the parameters from that file. 
Those settings *override* any equivalent settings contained in the python file, so if you are changing parameters in 
the python code and nothing is happening, check the json file (it took me a while to figure that out)

The optimisation run does not always produce better results, so look carefully). 
I tend to open the json file before running the optimisation, copy the contents to the paste buffer, and then paste them back if there was no improvement.

### Stoploss
Just a note (again) that I typically do _not_ optimise for stoploss - I just set it to a fixed number (usually 10%).
If you do optimise it, you will see better results in backtesting. However, my (hard-earned) experience is that this 
does not transfer to the real world - what happens is that, with a larger stoploss, one losing trade will typically 
wipe out gains from 10 or more winning trades. So, just cut your losses at 10% and move on.

## Hyperopt Loss Functions
freqtrade provides several loss functions for use with hyperopt. Each loss function will evaluate the results from 
running the strategy and assign a score based on those results (using profit etc). The hyperopt function basically starts 
off with several random combination of input parameters then tweaks them to try and produce lower results from the loss 
function (the convention is that lower is better).

_\<loss algorithm\>_ is one of:

- ShortTradeDurHyperOptLoss
- OnlyProfitHyperOptLoss
- SharpeHyperOptLoss
- SharpeHyperOptLossDaily
- SortinoHyperOptLoss
- SortinoHyperOptLossDaily

Run them in that order and see if you get different results. SharpeHyperOptLoss is a good starting point.

I did find that the built-in loss functions had problems dealing with cases where there are a very low number of trades 
with a high average win rate and/or average profit. To help address that, I wrote some custom loss functions, which can 
be found in the _hyperopts_ directory. <br>
To use them, you have to copy them to the _freqtrade/user\_data/hyperopts directory 
(which is outside this repository), and then specify one of them using the _-l_ or _--hyperopt-loss_ options.

For example:

>bash user_data/strategies/scripts/hyp_strat.sh -l WeightedProfitHyperoptLoss binance FBB_ROI

The available custom loss functions are:

| Loss Function               | Description |
|-----------------------------|------------------------------------------|
| ExpectancyHyperOptLoss      | Optimises based primarily on Expectancy (projected profit per trade) |
| PEDHyperOptLoss             | Optimises based equally on Profit, Expectancy and Duration |
| QuickHyperOptLoss           | Optimises based primarily on average duration of trades (shorter is better) |
| WinHyperOptLoss             | Optimises based primarily on Win/Loss ratio |
|  WeightedProfitHyperOptLoss | Optimises based primarily on profit |

All of these functions take multiple parameters into account, they just use different weightings. 
They also require a minum profit, number of trades and win/loss ratio.

The best general purpose loss function is _WeightedProfitHyperOptLoss_, and this is the default used by the 
_hyp\_strat.sh_ script

## Plotting Results

It is often very useful to see a visual representation of your strategy. You can do this using the plot-dataframe command:

>freqtrade plot-dataframe --strategy-path _\<path\>_ --strategy _\<strategy\>_  -p BCH/USD --timerange=_\<timerange\>_ --indicators1 ema5 ema20 --indicators2 mfi

This example creates a plot for pair BCH/USD, adds ema5 and ema20 to the main chart and adds a plot of mfi below that (these must exist in your dataframe for the strategy). You can find it in the _user_data/plot/_ directory, and in this case the name of the file would be: 
_user_data/plot/freqtrade-plot-BCH_USD-5m.html_
<p>You can open this in a browser, and you can zoom in/out and move around</p>
Any buys and sells generated by backtesting (if any) will be shown on the plot, which is very useful for debugging.

Tip: constrain the plot to a day or 2, otherwise you have to zoom in a lot. Also, when debugging, use the -p option in backtesting, which will cause the strategy to run only on the pair that you specify, and not across the whole list of pairs.
Also, choose the pair that has the worst performance in your strategy to see where things are going wrong.

## Dry Runs

Easy. In a command window, just run:

>freqtrade trade --dry-run --strategy-path _\<path\>_ --strategy _\<strategy\>_

If you install freqUI (instructions [here](https://www.freqtrade.io/en/stable/rest-api/)), then you can 
monitor the trades on a web page at http://127.0.0.1:8080/ (or whatever address you specify in config.json)

The script helper is:

>bash user_data/strategies/scripts/dryrun_strat -p _\<port\>_ _\<exchange\>_ _\<strategy\>_

The -p is optional, but if you want to run multiple strategies on the same exchange you need to use this. 
In such cases, there needs to be a matching config file in the base _freqtrade_ directory of the form _config\_\<exchange\>\_\<port\>.json_,
and the port number in the config file must match.

## Live Trading

For this, you need an account on an exchange, and you need to modify config.json to specify which exchange, trading limits, private keys etc.
You need to get API keys from the exchange website, but that is usually easy.

In a command window, just run:

>freqtrade trade --strategy _\<strategy\>_

The script helper is:

>bash user_data/strategies/scripts/run_strat -p _\<port\>_ _\<exchange\>_ _\<strategy\>_

You can monitor trades from the UI (see above), and from the exchange website/app

Note that you need your computer synched up to an NTP time source to do this.
