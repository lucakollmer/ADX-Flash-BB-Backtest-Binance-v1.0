# %% Preamble
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

__version__ = '0.1'
__author__ = 'Luca Kollmer'

# %% Import Packages

import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine

# pip install python-binance
from binance.client import Client

# Initialise Binance API Client.
client = Client()

# Set symbols to backtest strategy on.
coins = ('BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','ADAUSDT','XRPUSDT','DOTUSDT',
         'LUNAUSDT', 'DOGEUSDT','AVAXUSDT','SHIBUSDT','MATICUSDT','LTCUSDT',
         'UNIUSDT','ALGOUSDT','TRXUSDT', 'LINKUSDT','MANAUSDT','ATOMUSDT',
         'VETUSDT'
         )

# %% Request Price Data from Binance API and Transform to OHLC.

def get_minute_data(symbol, lookback):
    '''
    Request one minute price data from Binance API over 'lookback' days.
    
    Parameters
    ----------
    symbol : str
    lookback : str
        Lookback period in days.

    Returns
    -------
    df : TYPE
        OHLC dataframe.

    '''
    
    df = pd.DataFrame(
            client.get_historical_klines(
                symbol, '1m', lookback + ' days ago UTC'))
    # Remove volume column
    df = df.iloc[:,:5]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close']
    df[['Open','High','Low','Close']] = df[['Open',
                                            'High',
                                            'Low',
                                            'Close'
                                            ]].astype(float)
    df.Time = pd.to_datetime(df.Time, unit='ms')
    
    return df

# %% Compute True Range Indicator on HLC Series.

def get_true_range(high, low, close):
    '''
    Calculates True Range for HLC price series.

    Parameters
    ----------
    high : float
    low : float
    close : float

    Returns
    -------
    df : TYPE
        One column dataframe containing True Range price series.

    '''
    
    high_to_low = pd.DataFrame(high - low)
    close_to_high = pd.DataFrame(abs(high - close.shift(1)))
    close_to_low = pd.DataFrame(abs(low - close.shift(1)))
    
    true_range = [high_to_low, 
          close_to_high,
          close_to_low,
          ]
    
    df = pd.concat(true_range,
                   axis=1,
                   join='inner',
                   ).max(axis=1)
    
    return df

# %% Compute ATR, DMI and BB Indicators on OHLC Series.

def get_technicals(df,
                   timeframe,
                   bb_period,
                   adx_period):
    '''
    Computes Average True Range (ATR), Directional Movement Index (DMI) and
    Bollinger Band (BB) technical indicators to generate ADX Flash and buy/sell
    signals columns of a price series.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    timeframe : ?
        Price timeframe in minutes to transform one minute data into.
    bb_period : int
        Smoothing period of BB.
    adx_period : int
        Smoothing period of DMI.

    Returns
    -------
    df : TYPE
        Price series with ADX Flash signals and BB buy/sell signals.

    '''
    
    df = df.copy()
    
    # Transform series to new timeframe in minutes.
    #df = df.resample(timeframe).ffill()
    df.dropna(inplace=True)
    
    # Compute Bollinger Bands (BB) Indicator.
    df['Upper BB'] = df.Close.rolling(bb_period).mean() \
                     +  df.Close.rolling(bb_period).std(ddof=0) * 2
    df['Lower BB'] = df.Close.rolling(bb_period).mean() \
                     - df.Close.rolling(bb_period).std(ddof=0) * 2
                    
    # Create buy/sell signal when price closes outside BB.
    df['BBSignal'] = np.where(df.Close > df['Upper BB'], -1, 0)
    df['BBSignal'] = np.where(df.Close < df['Lower BB'], 1, df['BBSignal'])
    
    # Compute Directional Movement Indicator. #
    # Compute Average True Range.
    df['ATR'] = pd.DataFrame(
                    get_true_range(df.High, 
                                   df.Low, 
                                   df.Close)
                    ).ewm(alpha=(1/adx_period), 
                          min_periods=adx_period, 
                          adjust=False).mean()
                          
    # Compute Positive Directional Movement.
    df['Plus DM'] = np.where(df.High.diff() > -df.Low.diff(), 
                             df.High.diff(), 
                             0)
    df['Plus DM'] = np.where(df['Plus DM'] > 0, df['Plus DM'], 0)
    
    # Compute Negative Directional Movement.
    df['Minus DM'] = np.where(-df.Low.diff() > df.High.diff(), 
                              -df.Low.diff(), 
                              0)
    df['Minus DM'] = np.where(df['Minus DM'] > 0, df['Minus DM'], 0)  
    
    # Compute Positive and Negative Directional Index.
    df['Plus DI'] = (100*(df['Plus DM'].ewm(alpha=(1/adx_period), 
                                            min_periods=adx_period,
                                            adjust = False
                                            ).mean())) / df['ATR']
    df['Minus DI'] = (100*(df['Minus DM'].ewm(alpha=(1/adx_period),
                                              min_periods=adx_period, 
                                              adjust = False
                                              ).mean())) / df['ATR']
    
    # Compute Average Directional Index.
    df['DX'] = df['Plus DI'] + df['Minus DI']
    df['DX'] = np.where(df['DX'] == 0, 1, df['DX'])
    df['Raw ADX'] = 100*abs(df['Plus DI'] - df['Minus DI']) / df['DX']
    df['ADX'] = df['Raw ADX'].ewm(alpha=(1/adx_period), 
                                  min_periods=adx_period, 
                                  adjust = False).mean()
    
    # Generate ADX Flash signal when ADX is below 12.
    df['Trend'] = np.where(df['Plus DI'] > df['Minus DI'], 1, -1)
    df['Flash'] = np.where(df['ADX'] < 12, 1, 0)
    df['ADXSignal'] = np.where(df['Flash'].diff() == -1, 1, 0)    
    
    # Drop unneccesary data.
    df = df.drop(columns=['Upper BB', 'Lower BB', 'ATR', 'Plus DM', 'Minus DM',
                          'Plus DI', 'Minus DI', 'DX', 'Raw ADX'])

    return df

# %% Perform Backtest.

def analysis(df, timeframe, bb_period, adx_period, grace_period, fee, 
             position_size, stop, risk):
    '''
    Performs backtest of ADX Flash BB Strategy (Long Only).
    When ADX crosses 12 from below, enter long on close below BB, until ADX
    drops below 12 again, or price closes above open on bar ADX crosses 12.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    timeframe : str
        Transform price series from 1m to timeframe e.g. '3min'.
    bb_period : int
        Smoothing period for Bolling Bands.
    adx_period : int
        Smoothing period for Directional Movement Index.
    grace_period : int
        Bars after ADX crosses 12 from below prohibited from opening trades.
    fee : float
        Commission per transation e.g. 0.001.
    position_size : float
        Depracated.
    stop : float
        Distance from ADX open to set Stop Sell order.
    risk : float
        Depracated.

    Returns
    -------
    adx : TYPE
        DESCRIPTION.

    '''
    
    # Call get_technicals on price series.
    # First 200 rows contain inaccurate rolling averages so are dropped.
    df = get_technicals(df, timeframe, bb_period, adx_period).iloc[200:, :]
    
    # Calculate asset return assuming buy and hold from first row.
    df['Return'] = np.log(df.Close.pct_change() + 1)
    
    # Create dataframe to track ADX Flash signals.
    adx = pd.DataFrame(columns=['Time', 'Age', 'Flash', 'Status', 'Move', 
                                'Entries', 'Entry'])
    
    # Iterate row-wise through price series.
    for row in tqdm(df.itertuples()):
        
        # Iterate through adx to check Status, increment Age and update Move.
        for row2 in adx.itertuples():
            
            if (adx.size > 0):
                
                if (row2.Status == 0):
                    
                    # Increment Age of flash.
                    adx.at[row2.Index, 'Age'] += 1
                    
                    # Update maximum return achievable by flash.
                    if (row.High < row2.Flash):
                        adx.at[row2.Index, 'Move'] = min([(
                            row.Low - row2.Flash) / row2.Flash, row2.Move])
                    
                    elif (row.Low > row2.Flash):
                        adx.at[row2.Index, 'Move'] = max([(
                            row.High - row2.Flash) / row2.Flash, row2.Move])  
                    
                    # Check if price closes above/below Flash open price.
                    if (row2.Age > grace_period and not 
                        (row.High < row2.Flash or row.Low > row2.Flash)):
                        adx.at[row2.Index, 'Status'] = 1
                    
                    #C heck if Stop is hit.
                    elif (row.Low < row2.Flash * (1 - stop) or 
                          row.High > row2.Flash * (1 + stop)):
                        adx.at[row2.Index, 'Status'] = -1    
                        
        # Create new Flash in ADX when ADX Signal == 1.
        if (row.ADXSignal == 1):
            newdf = pd.DataFrame({'Time':[row.Index],
                                  'Age':[0],
                                  'Flash':[row.Open],
                                  'Status':[0],
                                  'Move':[0],
                                  'Entries':[0],
                                  'Entry':[0]}
                                 )
            adx = adx.append(newdf, ignore_index = True)       
            
        # Add entries to active flash.
        elif (adx.size > 0):
            x = len(adx.index) - 1 # Index of most recent Flash.
            positions = abs(adx.loc[adx['Status'] == 0, 'Entries']).sum()
            
            # Check for active flash and available .
            if adx.iloc[-1].Status == 0 and positions*position_size < 1:
                
                # Long Entry
                if (abs((row.Close - adx.at[x, 'Flash']) / adx.at[x, 'Flash'])
                    > fee * 2 and
                    (((stop - ((adx.at[x, 'Flash'] - adx.at[x, 'Entry']) 
                    / adx.at[x, 'Flash'])) * -adx.at[x, 'Entries'])
                    - abs(adx.at[x, 'Entries'])*2*fee)*position_size < risk):
                                        
                    if (row.BBSignal == 1 and 
                        adx.iloc[-1].Move < 0 or 
                        row.BBSignal == -1 and 
                        adx.iloc[-1].Move > 0):
                        
                        adx.at[x, 'Entry'] = ((adx.at[x, 'Entry'] \
                                             * abs(adx.at[x, 'Entries'])) \
                                             + row.Close) \
                                             / (abs(adx.at[x, 'Entries']) \
                                             + 1)
                        adx.at[x, 'Entries'] += row.BBSignal
    
    # Calculate Return.               
    adx['Return'] = np.where(adx['Status'] == 1, 
                             ((adx['Flash'] - adx['Entry']) \
                              / adx['Flash']) * adx['Entries'], 
                             0)
        
    adx['Stop'] =  np.where(adx['Entries'] != 0,
                            np.where(adx['Move'] < 0, 
                                     adx['Flash']*(1 - stop),
                                     adx['Flash']*(1 + stop)), 
                            0)
    
    adx['Return'] = np.where(adx['Status'] == -1,
                            (stop - ((adx['Flash'] - adx['Entry']) \
                                     / adx['Flash'])) * -adx['Entries'],
                            adx['Return'])

    # Calculate commission costs and Profit/Loss.
    adx['Cost'] = abs(adx['Entries']) * 2 * fee
    adx['PNL'] = (adx['Return']  - adx['Cost']) * position_size
    print(adx['PNL'].sum())
    
    # Compute Risk undertaken.
    adx['Risk'] = (((stop - ((adx['Flash'] - adx['Entry']) \
                  / adx['Flash'])) * -adx['Entries']) \
                  -abs(adx['Entries'])*2*fee)*position_size
        
    #Return dataframe of Flashes: adx.
    return adx

# %% Create Database of One Minute Price Series For List of Symbols.

minute_data = create_engine('sqlite:///Coin Minute Data.db')

for coin in tqdm(coins):
    get_minute_data(coin, '1').to_sql(coin, minute_data, index=False)
    
# %% Backtest Strategy For All Symbols in Database.

for coin in coins:
    df = pd.read_sql(coin, minute_data).set_index('Time')
    timeframe = '1min'
    print(coin)
    
    analysis(df,
             '1min', 
             20, 
             14, 
             7, 
             0.001, 
             0.05, 
             0.02, 
             0.01)
    
















