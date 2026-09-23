# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_accurate_momentum(
        ticker: str,
        interval: str, 
        change_period: int,
        in_cond: float, 
        take_profit: float, 
        stop_loss: float
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of profits per trade, returns per trade, total profit, cumulative return, optional plot
    '''

    df = raw_data[ticker]
    df = data_prep_accurate(df=df, interval=interval, change_period=change_period)
    trades, dates, n_ambiguous = accurate_momentum(df=df, in_cond=in_cond, take_profit=take_profit, stop_loss=stop_loss)

    return df, trades, dates, n_ambiguous
# %%
def data_prep_accurate(
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


def accurate_momentum(
        df: pd.DataFrame, 
        in_cond: float, 
        take_profit: float, 
        stop_loss: float,
    ) -> tuple:
    '''
    Input: Dataframe, momentum condition to enter a trade, take_profit and stop_loss conditions
    Output: A tuple of (price_in, price_out) for each trade entered, respective returns and dates (date_in, date_out)
    '''

    in_trade = False
    trades = []
    dates = []
    n_ambiguous = 0
    n = len(df)
    o = df['Open'].to_numpy()
    c = df['Close'].to_numpy()
    h = df['High'].to_numpy()
    l = df['Low'].to_numpy()
    pct = df['pct_change'].to_numpy()

    for i in range(n):
        if in_trade: 
            if o[i] >= tp or o[i] <= sl:
                # Case 1: the market jumped over the barrier since last available ticker
                out_price = o[i] 
            elif h[i] >= tp and l[i] <= sl:
                # Case 2: both barriers were touched in the same bar, dont know which first (low -> high vs high -> low), take pessimistic
                out_price = sl
                n_ambiguous += 1
            elif h[i] >= tp:
                out_price = tp
            elif l[i] <= sl:
                out_price = sl
            else:
                continue

            in_trade = False
            trades.append((in_price, out_price))
            dates.append((df.index[entry_i], df.index[i]))

        # Can buy in on same bar as sale        
        if np.isnan(pct[i]) or i + 1 == n or pct[i] < in_cond:
            continue

        in_trade = True
        entry_i = i + 1
        in_price = o[entry_i]

        tp = in_price * take_profit
        sl = in_price * stop_loss
    
    return trades, dates, n_ambiguous

       
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
        master_accurate_momentum(ticker = ticker, **hyperparams)


