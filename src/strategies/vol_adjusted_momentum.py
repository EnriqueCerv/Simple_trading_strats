# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_vol_adjusted_momentum(
        ticker: str,
        interval: str, 
        change_period: int,
        z_ind: float, 
        tp_sigma: float, 
        sl_sigma: float, 
        vol_window: int = 576,
        tau_mult: float = 3.0,
        min_sigma: float = 0.0,
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of profits per trade, returns per trade, total profit, cumulative return, optional plot
    '''

    df = raw_data[ticker]
    df, n_bars = data_prep_vol(df=df, interval=interval, change_period=change_period, vol_window=vol_window)
    trades, dates, n_ambiguous = vol_adjusted_momentum(df=df, z_ind=z_ind, tp_sigma=tp_sigma, sl_sigma=sl_sigma,
                                                       n_bars=n_bars, tau_mult=tau_mult, min_sigma=min_sigma)

    return df, trades, dates, n_ambiguous
# %%
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


def data_prep_vol(
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


def vol_adjusted_momentum(
        df: pd.DataFrame,
        z_ind: float, 
        tp_sigma: float, 
        sl_sigma: float,
        n_bars: int,
        tau_mult: float = 3.0,
        min_sigma: float = 0.0
    ) -> tuple:
    '''
    Input: Dataframe, z-score condition to enter a trade, take_profit and stop_loss sigma multipliers
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
    sigma_l = df['sigma_lookback'].to_numpy()
    z = df['z'].to_numpy()

    # expected time to touch a k-sigma barrier scales as k^2 under a driftless diffusion
    max_bars = int(np.ceil(tau_mult * tp_sigma ** 2 * n_bars))

    for i in range(n):
        if in_trade:
            exit_i = i

            if o[i] >= tp or o[i] <= sl:
                out_price = o[i]
            elif h[i] >= tp and l[i] <= sl:
                out_price = sl
                n_ambiguous += 1
            elif h[i] >= tp:
                out_price = tp
            elif l[i] <= sl:
                out_price = sl
            elif i - entry_i >= max_bars:
                if i + 1 == n:
                    break
                exit_i = i + 1
                out_price = o[exit_i]
            else:
                continue

            in_trade = False
            trades.append((in_price, out_price))
            dates.append((df.index[entry_i], df.index[exit_i]))

        if (np.isnan(z[i]) or i + 1 == n or z[i] < z_ind
                or not np.isfinite(sigma_l[i]) or sigma_l[i] <= min_sigma):
            continue
        
        in_trade = True
        entry_i = i + 1
        in_price = o[entry_i]
        sigma_freeze = sigma_l[i]

        tp = in_price * (1.0 + sigma_freeze * tp_sigma)
        sl = in_price * (1.0 - sigma_freeze * sl_sigma)
            
    
    return trades, dates, n_ambiguous

       
# %%

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'change_period': 240,
        'z_ind': 1.75, 
        'tp_sigma': 1.4, 
        'sl_sigma': 1.4, 
        'vol_window': 576,
        'tau_mult': 3.0,
        'min_sigma': 0.0
            }
    for ticker in tickers:
        master_vol_adjusted_momentum(ticker=ticker, **hyperparams)
