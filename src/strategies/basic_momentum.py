# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_basic_momentum(
        ticker: str,
        interval: str, 
        change_period: int,
        in_cond: float, 
        take_profit: float, 
        stop_loss: float
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of dataframe, trade tuples (price_in, price_out) and respective dates
    '''

    df = raw_data[ticker]
    df = data_prep_basic(df=df, interval=interval, change_period=change_period)
    trades, dates = basic_momentum(df=df, in_cond=in_cond, take_profit=take_profit, stop_loss=stop_loss)

    return df, trades, dates
# %%
def data_prep_basic(
        df: pd.DataFrame,  
        interval: str, 
        change_period: int
    ) -> pd.DataFrame:
    '''
    Input: DataFrame of a single ticker, ticker frequency, lookback window
    Output: Same dataframe with lagged returns over last (change_period / interval) tickers 
    (Best suited for tickers with no session breaks as lag is ticker based not time based)    
    '''

    interval = int(interval[:-1])
    n_bars = change_period // interval

    new_df = df.copy()
    new_df['pct_change'] = new_df['Close'].pct_change(n_bars)

    return new_df


def basic_momentum(
        df: pd.DataFrame, 
        in_cond: float, 
        take_profit: float, 
        stop_loss: float,
    ) -> tuple:
    '''
    Input: Dataframe, momentum condition to enter a trade, take_profit and stop_loss conditions
    Output: A tuple of (price_in, price_out) for each trade entered, respective dates (date_in, date_out)
    '''

    in_trade = False
    trades = []
    dates = []

    for idx in df.dropna(subset=['pct_change']).index:
        cur_price = df['Close'].loc[idx]
        if not in_trade:
            if df['pct_change'].loc[idx] >= in_cond:
                in_trade = True
                in_price = cur_price
                in_date = idx

        else:
            up_cond = cur_price > in_price * take_profit
            down_cond = cur_price < in_price * stop_loss

            if up_cond or down_cond:
                in_trade = False
                out_date = idx
                trades.append((in_price, cur_price))
                dates.append((in_date, out_date))
    
    return trades, dates
       
# %%

if __name__ == '__main__':

    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'change_period': 240,
        'in_cond': 0.01, 
        'take_profit': 1.01, 
        'stop_loss': 0.99
            }
    for ticker in tickers:
        master_basic_momentum(ticker = ticker, **hyperparams)
