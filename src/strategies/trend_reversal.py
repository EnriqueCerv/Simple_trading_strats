# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_trend_reversal(
        ticker: str,
        interval: str, 
        fast_window: int,
        slow_window: int,
        in_cond: float,
        out_cond: float,
        burn_spans: int = 3
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy. Interval is string ('5m') 
           Windows are in minutes.
           In and out_cond are thresholds for difference in ewma averages (in at >0, out at <0 eg)
    Output: Tuple of dataframe, trade tuples (price_in, price_out) and respective dates
    '''

    df = raw_data[ticker]
    df = data_prep_trend_reversal(df=df, interval=interval, fast_window=fast_window, slow_window=slow_window, burn_spans=burn_spans)
    trades, dates = trend_reversal(df=df, in_cond=in_cond, out_cond=out_cond)

    return df, trades, dates
# %%
def bars_in_period(interval: str, window: int) -> int:
    '''
    Input: bar interval string ('5m', '1h', '1d'), lookback window in MINUTES
    Output: number of bars spanning that window
    '''

    unit = interval[-1].lower()
    value = int(interval[:-1])

    minutes_per_unit = {'m': 1, 'h': 60, 'd': 1440}
    if unit not in minutes_per_unit:
        raise ValueError(f'unsupported interval unit: {interval!r}')

    return window // (value * minutes_per_unit[unit])

def ewma_lambda(span: int) -> float:
    '''
    Returns centre of mass of ewma
    '''
    return (span - 1) / (span + 1)

def kernel_l2_norm(lambda_fast: float, lambda_slow: float) -> float:
    # sum_j (lambda_slow^{j+1} - lambda_fast^{j+1})^2
    return (
        lambda_slow**2 / (1 - lambda_slow**2)
        - 2 * lambda_slow * lambda_fast / (1 - lambda_slow * lambda_fast)
        + lambda_fast**2 / (1 - lambda_fast**2)
        )


def data_prep_trend_reversal(
        df: pd.DataFrame,  
        interval: str, 
        fast_window: int,
        slow_window: int,
        burn_spans: int = 3
    ) -> tuple:
    '''
    EWMA-crossover trend signal on log prices for a single ticker.

    Adds columns:
        D       : fast - slow EWMA of log price (weighted sum of past log returns)
        sigma   : EWMA vol of 1-bar log returns
        mu_hat  : D / (L_slow - L_fast), local drift estimate per bar
        D_norm  : D / (sigma * ||w||_2), ~N(0,1) under no-drift iid null

    Signal at bar t uses data up to and including Close_t;
    trade on bar t+1 in the backtest.
    Bar-count based, so best for tickers without session breaks.
    '''

    fast_bars = bars_in_period(interval=interval, window=fast_window)
    slow_bars = bars_in_period(interval=interval, window=slow_window)

    new_df = df.copy()
    log_price = np.log(new_df['Close'])
    log_returns = log_price.diff()

    fast_ewma = log_price.ewm(span=fast_bars, adjust=False).mean()
    slow_ewma = log_price.ewm(span=slow_bars, adjust=False).mean()
    diff = fast_ewma - slow_ewma

    lambda_fast, lambda_slow = ewma_lambda(fast_bars), ewma_lambda(slow_bars)
    kernel = (slow_bars - fast_bars) / 2 # This is Ls - Lf
    kernel_l2 = kernel_l2_norm(lambda_fast, lambda_slow)
    sigma = log_returns.ewm(span=slow_bars, adjust=False, min_periods=slow_bars).std()

    new_df['D'] = diff
    new_df['sigma'] = sigma
    new_df['D_norm'] = diff / (sigma * kernel_l2)
    new_df['mu_hat'] = diff / kernel

    burn = burn_spans * slow_bars
    new_df.loc[new_df.index[:burn],  ['D', 'mu_hat', 'D_norm']] = np.nan

    return new_df
    

def trend_reversal(
        df: pd.DataFrame,
        in_cond: float = 1.0,
        out_cond: float = 0.0
    ) -> tuple:
    '''
    Long EWMA-crossover strategy with bands.
    Enter when D_norm >= in_cond, exit when D_norm <= out_cond.
    Decisions use D_norm at the close of bar i and execute at the open of bar i+1.

    Returns:
        trades     : list of (price_in, price_out)
        dates      : list of (date_in, date_out)
        open_trade : (price_in, last_close, date_in, last_date) if still in a trade, else None
    '''

    in_trade = False
    trades = []
    dates = []
    n = len(df)
    o = df['Open'].to_numpy()
    d_norm = df['D_norm'].to_numpy()

    for i in range(n - 1):
        if in_trade:
            if d_norm[i] > out_cond:
                continue

            in_trade = False
            out_price = o[i + 1]
            trades.append((in_price, out_price))
            dates.append((df.index[entry_i], df.index[i + 1]))

        if np.isnan(d_norm[i]) or d_norm[i] < in_cond:
            continue
        
        in_trade = True
        entry_i = i + 1
        in_price = o[entry_i]
            
    
    return trades, dates

import numpy as np
import pandas as pd


def in_cond_floor(
        kappa: float,
        fast_bars: int,
        slow_bars: int,
        sigma: float | pd.Series,
        hold_bars: float = 51,
        shrink: float = 1.0,
    ) -> float | pd.Series:
    '''
    Minimum D_norm entry threshold for expected captured drift to cover round-trip cost.

    kappa     : round-trip cost in log-return units (e.g. 10 bps total -> 0.001)
    sigma     : per-bar log-return vol (scalar, or the 'sigma' column for a time-varying floor)
    hold_bars : expected holding time in bars (e.g. median from a backtest)
    shrink    : fraction of the estimated drift you expect to actually realise (0 < shrink <= 1)
    '''
    fast_bars, slow_bars = bars_in_period('5m', window=fast_bars), bars_in_period('5m', window=slow_bars)
    lam_f, lam_s = ewma_lambda(fast_bars), ewma_lambda(slow_bars)
    w_norm = np.sqrt(kernel_l2_norm(lam_f, lam_s))
    lag_diff = (slow_bars - fast_bars) / 2
    return kappa * lag_diff / (shrink * sigma * w_norm * hold_bars)
       
# %%

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'fast_window': 60,
        'slow_window': 180,
        'in_cond': 1.0,
        'out_cond': 0.0,
        'burn_spans': 3
            }
    for ticker in tickers:
        _, _, dates = master_trend_reversal(ticker=ticker, **hyperparams)
    diffs = np.array([(t2 - t1).total_seconds() / 60 // 5 for t1, t2 in dates])
    median = np.median(diffs) # 51