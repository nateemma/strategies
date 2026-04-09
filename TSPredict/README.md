# Time Series Prediction (TSPredict)

_*NOTE: these streategies currently do not work too well. Some of them perform slightly better than Market, but not much more.  
The fundamental problem appears to be that the regression algorithms cannot handle the complexity of the underlying time series (predominantly the gain). I am currently investigating ways to address this*_

_Also, the current performance is worse than previous releases because I fixed a subtle bug involving lookahead bias_

Strategies in this directory implement a common approach to 'simply' estimating future values of a time series - in this case, the gain. I use gain instead of price because the values are similar across all pairs (price is not at all) and so I can create models that work for any pair.

In general, most (all except TS_Wavelet*) of these strategies will try to load an existing model, but if it is not found it will create and train a new model. So, the first time you run a strategy, do so over a long period of time.
On subsequent runs, the strategy will load the saved model, and use that as the basis for all pairs. Pair-specific data is used to incrementally train the model with new data as it becomes available (but this is not saved). In theory, prediction accuracy should get better over time

_*Warning*_: backtest results can look really good for these strategies, but be sure to try these out in a dry run.  
The backtest results are good because the trades are 'perfect', meaning that they execute immediately; in the real world, there are delays and spreads that take away profit from each trade, and since these strategies inherently look at quick in & out trades that has a big impact. 
Also, you can hyperopt these strategies to get what appears to be really good returns, usually by emphasising fast trades. In live environments, this approach tends to get hit hard by quick downward swings and reversals.  
So, I have been trying to tune the hyperparameters to be 'safer', and optimise for expectancy rather than profit.

_*
Currently, the win rate on these strategies is pretty good, but they all seem to suffer from buying into 'false' reversals, i.e. the price is going down then swings up, but then quickly drops (a long way) down again. The strategies see this reversal and trade into it. I am currently looking into various momentum indicators to try and fix this...
*_

## Quick Start

1. Download data<br>

    > zsh user_data/strategies/scripts/download.sh binanceus

2. Test over a long period of time to create a decent base model (only need to do this once per strat). No need for this with Wavelet strategies.

    > zsh user_data/strategies/scripts/test_strat.sh -n 600 TSPredict \<_strat_\>

3. Test over reasonable time period

    > zsh user_data/strategies/scripts/test_strat.sh -n 30 TSPredict \<_strat_\>

4. Plot data

    > zsh user_data/strategies/scripts/plot_strat.sh -n 7 TSPredict \<_strat_\> \<_pair_\>

5. Examine plot
In a browser, open the file file://user_data/plot/freqtrade-plot-ALGO_USDT-5m.html <br>
but you probably need to fix the path to point to wherever you installed freqtrade. 
Replace ALGO_USDT with whatever pair you used when running plot_strat.sh

The plot for _predicted\_gain_ should look something like the plot for _gain_, but shifted forward in time (since it is attempting to predict future values of _gain_). When the value of _predicted\_gain_ crosses above _target\_profit_, an entry signal should be generated. Conversely, when the value of _predicted\_gain_ crosses below _target\_loss_, an exit signal should be generated. Note that this may not always happen, since extra guard conditions could be active (look in the .json file)

## Files

The majority of functional code is in the class _TSPredict.py_ - this implements the basic dataframe handling, model loading/saving/training, custom stoploss/exit and populate_entry/exit functions.<br>
Similarly _TS_Wavelet.py_ contains the base code for the Wavelet family of strategies (see below)

Other classes add on different approaches to modelling the future values, for example:

- TS_Gain_*.py<br>
These classes predict on _only_ the historical gain values. These are not particularly good strategies, they are mostly here to serve as a baseline (any viable strategy should do better than these), and a way to find lookahead bias.

- TS_Simple_*.py<br>
These add some additional indicators and estimate future gain using a variety of different algorithms (e.g. TS_Simple_SGD.py uses a Stochastic Gradient Descent algorithm)<br>
This is actually a good platform for testing whether indicators make a difference or not. Indicators that vary wildly or have discontinuities will likely mess up the predictions.
Note: if you want to try different regression/prediction algorithms, be aware that they really need to support incremental training, i.e. the model is updated with new data rather than completely retrained.

- TS_Coeff_*.py<br>
These add coefficients to the indicators that are based on various signal estimation algorithms, such as FFT, DWT etc. The estimation coefficients are added to the indicators and the prediction/regression algorithm is trained using those coefficients

- TS_Wavelet_*.py<br>
Instead of predicting gain using various signal estimation techniques or regression/prediction algorithms, these decompose the gain data into subcomponents using various transforms (DWT, SWT etc.), predict the future values of each subcomponent and then rebuilds the signal to create a a prediction of future gain. These are _very_ compute intensive, so I had to use a fast regression algorithm (PassiveAggressiveRegressor from sklearn). XGBoost is better, but does not run in real time when performing a dry run.<br>
Note: these strategies do not save/reload models, so you do not need to train them before use

Note: yes, I know there many other algorithms that could be used. If you don't see them here, it generally means they didn't perform well, so I didn't include them.

## Indicators

If you want to add indicators, then you should crate a new strategy that inherits from the strategy that you want to modify, and override the function add_strategy_indicators(). Then, follow the steps below to create a model for the new strategy (unless you are subclassing one of the TS_Wavelet strategies, which does not use stored models).

## Models

Models, if used, are saved in user_data/strategies/TSPredict/models/\<_classname_\>

If you change indicators or the prediction/regression algorithm (or sometimes update the package) then you will need to delete and regenerate the model.

For example, say we change something in TS_Simple_PA.py that requires a model update, we would probably do the following:

~~~
# remove existing model
rm -r user_data/strategies/TSPredict/models/TS_Simple_PA

# download data (this will only download missing data)
zsh user_data/strategies/scripts/download.sh -n 600 binanceus

# regenerate the model over a long time period
zsh user_data/strategies/scripts/test_strat.sh -n 600 TSPredict TS_Simple_PA

# test over a recent time period
zsh user_data/strategies/scripts/test_strat.sh -n 30 TSPredict TS_Simple_PA

# plot the last few days
zsh user_data/strategies/scripts/plot_strat.sh -n 3 TSPredict TS_Simple_PA ALGO/USDT

# load the file user_data/plot/freqtrade-plot-ALGO_USDT-5m.html into a browser and check buy/sell events

~~~

## Downloading Test Data

To run backtest and hyperopt, you need to download data to your local environment.
<br>
You do this using a command like this:

> freqtrade download-data --timerange=_\<timerange\>_ -c _\<config\>_-t 5m

I typically download for the past 180 days, which is the default used by the various scripts.

For convenience, you can also just run :

> zsh user_data/strategies/scripts/download.sh *\[\<exchange\>\]*

## Backtesting

Backtesting is where you run your strategy on historical data to see how it would have performed.

In all command/shell examples, _\<strategy\>_ is the name of the strategy being used, and _\<timerange\>_ is the range
of time to use for testing (duh).

Examples of _\<timerange\>_ might be:

- _20210501-_       May 1st, 2021 until present date
- _20210603-20210605_   June 3rd 2021 until June 5th 2021

Before you can backtest, you must download data for testing.

It is recommended that you update fairly often (e.g. once per week) and on a limited time range (e.g. the last month).
Optimising results for the past year or 6 months doesn't really help you perform better with the current market
conditions.

Backtesting can be done using the following command:

> freqtrade backtesting -c _\<config\>_ --strategy-path _\<path\>_ --strategy _\<strategy\>_  --timerange=
_\<timerange\>_

Or, you can use a script:

> zsh user_data/strategies/scripts/test_strat.sh _\<exchange\>_ _\<strategy\>_

Use the -h option for options.

*NOTE*: if you get really good results in backtesting (100% or more) then it is highly likely that your strategy is
performing lookahead (using future data). This is remarkably easy to do, since the entire set of test data is present in
the dataframe when your strategy is calculating indicators. So, if you do something like take a mean, find a min/max
etc. then you are operating on the entire data set, not just the older data. To avoid this, either use the TA-lib
functions or process the data through the _rolling_ mechanism.

See [here](https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies) for (a
little) more information

Also, some example strategies with subtle lookahead bias can be viewed (with an explanation)
at: <https://github.com/freqtrade/freqtrade-strategies/tree/master/user_data/strategies/lookahead_bias>

## Hyper-Parameter Optimisation

Run the hyperopt command to search for 'optimal' parameters in your strategy.

The parameters that can be tuned are defined in the strategy using calls such as:

> buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)

In this example, the code is telling freqtrade that the strategy has a _buy_ parameter named _buy\_bb\_gain_, which is a
floating point number (Decimal) between 0.01 and 0.1, using 2 decimal places

Note that performance is _dramatically_ affected by the parameters (especially ROI) and timeframe. Also, strategies
tuned for a specific exchange will typically not perform well on other exchanges.

The freqtrade command to run the optimisation is:

> freqtrade hyperopt -c _\<config\>_ --strategy-path _\<path\>_ --strategy _\<strategy\>_  --spaces _\<space\>_
> --hyperopt-loss _\<loss algorithm\>_   --timerange=_\<timerange\>_

where:

_\<space\>_ specifies which space to try and optimise. Typical options are buy, sell, roi, stoploss and trailing. I tend
to _not_ optimise for stoploss, and just set it manually to 10% (-0.1), or 99% for strategies that use dynamic ROI or a
custom stoploss (e.g. FBB_Solipsis)

Or, you can use a script:

> zsh user_data/strategies/scripts/hyp_strat.sh -s "buy sell roi" _\<exchange\>_ _\<strategy\>_

If you have a fast computer, try _--space buy sell_ and then _--spaces sell_, but if you have a slower computer, try
them in sequence.

NOTE: the optimised parameters are written to a json file that matches the strategy file, e.g. DWT.py will produce
DWT.json. freqtrade commands (backtesting, dryrun, live running) will take the parameters from that file. Those
settings _override_ any equivalent settings contained in the python file, so if you are changing parameters in the
python code and nothing is happening, check the json file (it took me a while to figure that out)

The optimisation run does not always produce better results, so look carefully). I tend to open the json file before
running the optimisation, copy the contents to the paste buffer, and then paste them back if there was no improvement.

## Plotting Results

It is often very useful to see a visual representation of your strategy. You can do this using the plot-dataframe
command:

> freqtrade plot-dataframe --strategy-path _\<path\>_ --strategy _\<strategy\>_  -p BCH/USD --timerange=_\<timerange\>_
> --indicators1 ema5 ema20 --indicators2 mfi

This example creates a plot for pair BCH/USD, adds ema5 and ema20 to the main chart and adds a plot of mfi below that (
these must exist in your dataframe for the strategy). You can find it in the _user_data/plot/_ directory, and in this
case the name of the file would be:
_user_data/plot/freqtrade-plot-BCH_USD-5m.html_
<p>You can open this in a browser, and you can zoom in/out and move around</p>
Any buys and sells generated by backtesting (if any) will be shown on the plot, which is very useful for debugging.

Tip: constrain the plot to a day or 2, otherwise you have to zoom in a lot. Also, when debugging, use the -p option in
backtesting, which will cause the strategy to run only on the pair that you specify, and not across the whole list of
pairs. Also, choose the pair that has the worst performance in your strategy to see where things are going wrong.

## Dry Runs

Easy. In a command window, just run:

> freqtrade trade --dry-run --strategy-path _\<path\>_ --strategy _\<strategy\>_

If you install freqUI (instructions [here](https://www.freqtrade.io/en/stable/rest-api/)), then you can monitor the
trades on a web page at <http://127.0.0.1:8080/> (or whatever address you specify in config.json)

The script helper is:

> zsh user_data/strategies/scripts/dryrun_strat.sh -p _\<port\>_ _\<exchange\>_ _\<strategy\>_

The -p is optional, but if you want to run multiple strategies on the same exchange you need to use this. In such cases,
there needs to be a matching config file in the base _freqtrade_ directory of the form _
config\_\<exchange\>\_\<port\>.json_, and the port number in the config file must match.

## Live Trading

Just to repeat:

**Never, _ever_, go to live trading without first running the strategy (for at least a week) in a dry run to see how it
performs with real, live data**

For this, you need an account on an exchange, and you need to modify config.json to specify which exchange, trading
limits, private keys etc. You need to get API keys from the exchange website, but that is usually easy.

In a command window, just run:

> freqtrade trade --strategy _\<strategy\>_

The script helper is:

> zsh user_data/strategies/scripts/run_strat.sh -p _\<port\>_ _\<exchange\>_ _\<strategy\>_

You can monitor trades from the UI (see above), and from the exchange website/app

Note that you need your computer synched up to an NTP time source to do this.
