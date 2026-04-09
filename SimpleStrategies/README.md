# Simple Strategies

These strategies are relatively simple strategies that take 'classic' buy/sell technical indicators and use hyperopt to 
select the parameters. I also add in a simple guard metric based on RMI to prevent 'silly' trades.

Essentially, I went through the indicators available in TA-lib and finta and implemented a strategy around all of the indicators that could be used for buy/sell decisions.

Note that I also set the freqtrade parameter exit_profit_only = True, which means that the strategies will hold on to trades for a very long time.

Performance is actually not bad. None of them beat the market in general (which is actually a good sign), but they are profitable.