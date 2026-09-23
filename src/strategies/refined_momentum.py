# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_refined_momentum(
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
    df = data_prep_refined(df=df, interval=interval, change_period=change_period)
    trades, dates = refined_momentum(df=df, in_cond=in_cond, take_profit=take_profit, stop_loss=stop_loss)

    return df, trades, dates

# %%
def data_prep_refined(
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


def refined_momentum(
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
    n = len(df)
    o = df['Open'].to_numpy()
    c = df['Close'].to_numpy()
    pct = df['pct_change'].to_numpy()

    for i in range(n):
        if not in_trade:
            if np.isnan(pct[i]) or i + 1 == n or pct[i] < in_cond:
                continue
            
            in_trade = True
            entry_i = i + 1
            in_price = o[entry_i]

        else:
            cur_close = c[i]
            up_cond = cur_close > in_price * take_profit
            down_cond = cur_close < in_price * stop_loss

            if up_cond or down_cond:
                if i + 1 == n:
                    break
                in_trade = False
                exit_i = i + 1
                trades.append((in_price, o[exit_i]))
                dates.append((df.index[entry_i], df.index[exit_i]))
    
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
        master_refined_momentum(ticker = ticker, **hyperparams)
