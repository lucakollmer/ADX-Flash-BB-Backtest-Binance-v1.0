# ADX-Flash-BB-Backtest-Binance-v1.0
First attempt at implementing a backtest of my ADX Flash BB Strategy.
For now only longs allowed.
When ADX crosses 12 from below, a Flash is recorded at the open price. 
After a grace_period, closes below the lower Bollinger Band create a buy signal.

Issues with code:
Tracking all Flashes in a Dataframe and iterating through all of them, even complete ones, is inefficient.
Does not model longs.
Did not get around to implementing position sizing or risk management, implemented in v2.0.
