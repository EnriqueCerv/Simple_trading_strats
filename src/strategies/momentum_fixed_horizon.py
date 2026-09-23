# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_momentum_fixed_horizon(
        ticker: str,
        interval: str, 
        change_period: int,
        z_in: float, 
        horizon: int
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of dataframe, trade tuples (price_in, price_out) and respective dates
    '''

    df = raw_data[ticker]
    df, _ = data_prep_fixed_momentum(df=df, interval=interval, change_period=change_period)
    trades, dates = momentum_fixed_horizon(df=df, z_in=z_in, horizon=horizon)

    return df, trades, dates
# %%
# def data_prep_fixed_momentum(
#         df: pd.DataFrame,  
#         interval: str, 
#         change_period: int
#     ) -> pd.DataFrame:
#     '''
#     Input: DataFrame of a single ticker, ticker frequency, lookback window
#     Output: Same dataframe with lagged returns over last (change_period / interval) tickers 
#     (Best suited for tickers with no session breaks as lag is ticker based not time based)    
#     '''

#     interval = int(interval[:-1])
#     n_bars = change_period // interval

#     new_df = df.copy()
#     new_df['pct_change'] = new_df['Close'].pct_change(n_bars)

#     return new_df

def bars_in_period(interval: str, change_period: int) -> int:
    '''
    Input: bar interval string ('5m', '1h', '1d'), lookback window in MINUTES
    Output: number of bars spanning that window
    '''

    unit = interval[-1].lower()
    value = int(interval[:-1])

    minutes_per_unit = {'m': 1, 'h': 60, 'd': 1440}
    if unit not in minutes_per_unit:
        raise ValueError(f'unsupported interval unit: {interval!r}')

    return change_period // (value * minutes_per_unit[unit])


def data_prep_fixed_momentum(
        df: pd.DataFrame,  
        interval: str, 
        change_period: int,
        vol_window: int = 576
    ) -> tuple:
    '''
    Input: DataFrame of a single ticker, ticker frequency, lookback window, vol estimation window
    Output: (dataframe with sigma_lookback and z columns, n_bars)
    (Best suited for tickers with no session breaks as lag is ticker based not time based)    
    '''

    n_bars = bars_in_period(interval=interval, change_period=change_period)

    new_df = df.copy()
    pct_change = new_df['Close'].pct_change()
    sig_bar = pct_change.rolling(vol_window).std()
    new_df['sigma_lookback'] = sig_bar * np.sqrt(n_bars)
    new_df['z'] = new_df['Close'].pct_change(n_bars) / new_df['sigma_lookback']

    return new_df, n_bars

def momentum_fixed_horizon(
        df: pd.DataFrame, 
        z_in: float,
        horizon: int,
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
    sigma_l = df['sigma_lookback'].to_numpy()
    z = df['z'].to_numpy()

    for i in range(n):
        if in_trade: 
            if i - entry_i == horizon:
                out_price = o[i]
                in_trade = False
                trades.append((in_price, out_price))
                dates.append((df.index[entry_i], df.index[i]))

        # Can buy in on same bar as sale        
        # if np.isnan(pct[i]) or i + 1 == n or pct[i] < z_in:
        #     continue

        # in_trade = True
        # entry_i = i + 1
        # in_price = o[entry_i]

        # Can buy in on same bar as sale        
        if (np.isnan(z[i]) or i + 1 == n or z[i] < z_in or not np.isfinite(sigma_l[i])):
            continue
        
        in_trade = True
        entry_i = i + 1
        in_price = o[entry_i]
    
    return trades, dates

       
# %%

if __name__ == '__main__':

    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'change_period': 240,
        'horizon': 20
            }
    for ticker in tickers:
        master_momentum_fixed_horizon(ticker = ticker, **hyperparams)


