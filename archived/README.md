# Phil's Custom freqtrade Crypto Trading Strategies (Archived)

*Note*: this folder conatins archived versions of strategies that I no longer support.
<br>
I leave them here because some of them are instructive for people new to writing freqtrade strategies 




This folder contains the code for a variety of custom trading strategies for use with the [freqtrade](https://www.freqtrade.io/) framework.

Other examples may be found in various github repositories:

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

Note: as of August 28th, 2021, I modified the code and scripts to support multiple exchanges. 
This means that there are some exchange-specific directories and files. 
If you only plan to use one exchange, then just set up your config.json file appropriately and modify the Config.py in 
the strategy directory to match your hyperopt results.
<br>
See the section on [Multiple Exchanges](multi-exchange-support) for more info

## Disclaimer

These strategies are for educational purposes only

Please note that the performance of some of these strategies is really bad (usually the most hyped ones).
I leave them here as templates for future work.

Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by testing strategies using backtesting then run the trading bot in Dry-run. 
Do not engage money before you understand how it works and what profit/loss you should expect. 
Also, do not backtest the strategies in a period of huge growth (like 2020 for example), 
when any strategy would do well. I recommend including May 2021, which includes a pretty big crash (-50%).

## List of Strategies

The following is a list of (most of) my custom strategies and a rough assessment of performance. 
They are listed in order of relative performance, good to bad, but please note that performance depends a lot upon the 
time period being tested, which exchange you are using, and which pairs you are trading.

Note: I am pursuing a "buy on dips, sell at a profit" approach, so most of these strategies
do not issue sell signals at all. Instead, they use the ROI and stoploss mechanisms of freqtrade to sell (hopefully at a profit).
I started off trying to issue sell signals, but performance was bad and I've had far better results just holding.

Also, I now use the same ROI/stoploss parameters across all strategies (they are defined in Config.py). 
This is because my goal is to combine the best individual strategies into a combined strategy (ComboHold.py)

In this context, the performance ratings are:

- Good: usually makes a profit
- OK: can make a slight profit or loss, depending upon the market conditions and pairs being traded
- Bad: usually loses money (often a lot)

|        Strategy | Performance |   Description                                 | 
|-----------------|-------------|-----------------------------------------------|
| ComboHold | Good |Combines the best of the strategies below|
| BigDrop | Good |Looks for a minimum %age total drop over a specified time|
| NDrop | Good |Looks for N consecutive 'drops', with a minimum %age total drop|
| NSeq | Good |Looks for N consecutive down candles followed by an up candle|
| BBBHold | Good |Buys when the close crosses below the lower Bollinger Band and holds|
| BTCNDrop | Good |Like BigDrop, but triggers on BTC/USD (the assumption is BTC/USD moves ahead of other pairs)|
| MACDCross | Good |Classic MACD crossing MACDSignal|
| BTCJump | Good |triggers on a big jump in BTC/USD (the assumption is BTC/USD moves ahead of other pairs)|
| BTCNDrop | Good |Like NDrop, but triggers on BTC/USD (the assumption is BTC/USD moves ahead of other pairs)|
| BTCNSeq | Good |Like NSeq, but triggers on BTC/USD (the assumption is BTC/USD moves ahead of other pairs)|
| EMABounce | Good |Looks for situations where the current price is a specified %age below the 20 day EMA, and the short term EMA is changing direction (up). Does not trigger often, but very reliable|
| Strategy003 | OK |This is from the freqtrade samples|
| BTCEMABounce | OK |Like EMABounce, but triggers on BTC/USD|
| Squeeze001 | OK |This is a variation of SqueezeMomentum from Lazy Bear, but with hyperopt-generated parameters. It works well, but I can't explain why!|
| BBKCBounce | OK |Buys when a candle crosses back above both the lower Bollinger and Keltner bands|
| BuyDips | OK |Buys on perceived dips and holds|
| KeltnerBounce | OK |Buy when price crosses lower Donchian band and hold|
| SimpleBollinger | OK |Buy when price crosses upper Bollinger band, sell when it crosses the lower band|
| SqueezeOff | OK |Buys when Squeeze Momentum transitions to 'off'|
| BollingerBounce | OK |Buy when price crosses lower Bollinger band and hold|
| Squeeze002 | OK |The classic LazyBear Squeeze Momentum|
| Patterns2 | OK |Looks for buy/sell candle patterns|
| ADXDM | Bad |Trade based on ADX value, plus DM+/-|
| DonchianChannel | Bad |Buy when price crosses upper Donchian band, sell when it crosses the lower band|
| DonchianBounce | Bad |Buy when price crosses lower Donchian band and hold|
| EMA50 | Bad |Classic fast and slow EMAs crossing |
| EMABreakout | Bad |Buy when price crosses above EMA|
| EMACross | Bad |Classic price crossing above/below EMA|
| KeltnerChannels | Bad |Buy when price crosses upper Keltner band, sell when it crosses the lower band|
| MACDTurn | Bad |Like MACDCross, but looks for a change in direction of the MACD Histogram to try and predict a trend before it happens (and fails miserably)|
| MFI2 | Bad |Buy/sell based on Money Flow Index (MFI)|
| SARCross | Bad |Classic price crossing above/below SAR|

## Setting Up Your Configuration

See the freqtrade docs, but basically, run the setup script and edit the config.json file to modify the list of pairs that you will use. Look at the website for the exchange you are using to se which pairs are supported (note that the default configuration generated is wrong)

## Backtesting

Backtesting is where you run your strategy on historical data to see how it would gave performed.

In all command/shell examples, _\<strategy\>_ is the name of the strategy being used, and _\<timeframe\>_ is the timeframe (duh).

Examples of _\<timeframe\>_ might be:
- _20210501-_       May 1st, 2021 until present date
- _20210603-20210605_   June 3rd 2021 until June 5th 2021

Before you can backtest, you must download data for testing. You do this using a command like this:

>freqtrade download-data  --timerange=_\<timerange\>_

It is recommended that you update fairly often (e.g. once per week) and on a limited time range (e.g. the last month). 
Optimising results for the past year or 6 months doesn't really help you perform better with the current market conditions.

Backtesting can be done using the following command:

>freqtrade backtesting --strategy _\<strategy\>_  --timerange=_\<timerange\>_

## Hyper-Parameter Optimisation

Run the hyperopt command to search for 'optimal' parameters in your strategy. 

Note that performance is _dramatically_ affected by the ROI parameters and timeframe. 

>freqtrade hyperopt --strategy _\<strategy\>_  --space _\<space\>_ --hyperopt-loss _\<loss algorithm\>_   --timerange=_\<timerange\>_

where:

_\<space\>_ specifies which space to try and optimise. It is recommended to try these combinations in order:

1. buy (using SharpeHyperOptLossDaily)
2. roi stoploss  (using OnlyProfitHyperOptLoss)
3. trailing  (using OnlyProfitHyperOptLoss)


If you try to optimise them all at once, you don't get good results. Also, not the use of different loss functions above


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

## Multi Exchange Support
Exchanges have different pairlists, different base currencies and different histories, so you really need different 
parameters for each exchange. The way I address this is by putting exchange-specific code in a subdirectory below the 
strategy directory whose name matches the exchange tag used by freqtrade (e.g. binanceus, kucoin, coinbasepro). 
<br>
If you look, you will find files such as Config.py in each subdirectory. These files contain exchange-specific code, 
which is usually the set of hyperparameters for the strategies rolled into ComboHold plus the ROI, stoploss and 
trailing parameters.

Because of this, you have to manage the _PYTHON_PATH_ environment variable or import statements will not find the 
correct  files (python is the only language I know of that doesn't have the concept of importing from the same 
directory, you have to know the structure relative to where the main code is running - which is a very bad design 
if you ask me).

To help with this. I added a bunch of shell scripts in the _user_data/strategies/scripts_ directory:

| Script | Description |
|-----------|------------------------------------------|
|testrat.sh|Usage: _bash teststrat.sh <exchange> <srategy> <args>_<br>Tests an individual strategy for the specified exchange. You can add additional parameters (for _freqtrade backtest_) in the 3rd argument (enclose in quotes if you need to use multiple")|
|hypstrat.sh||
|testall.sh||
|hyperall.sh||
|compareResults.sh||
|genParams.sh||
|overnight.sh||


Please note that these scripts all expect thre to be a config file for the exchange that is named in the form:  
_config\_<exchange>.json_ (e.g. _config_binanceus.json_)

Note: if you only use 1 exchange, then just use the Config.py that is in the main _user_data/strategies_ directory.