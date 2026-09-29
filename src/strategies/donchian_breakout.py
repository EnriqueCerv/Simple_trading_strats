# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_donchian_breakout(
        ticker: str,
        interval: str, 
        lookback: int,
        period: int,
        k: float, 
        chandelier: bool = False,
        intrabar: bool = False
    ):
    '''
    Long-only Donchian breakout with ATR trailing stop for a single ticker.

    Input:
        ticker     : key into raw_data
        interval   : bar size string ('5m', '1h', '1d')
        lookback   : Donchian channel window, in MINUTES
        period     : Wilder ATR period, in BARS
        k          : stop distance in ATR multiples
        chandelier : stop anchored to running high since entry (True) or to close (False)
        intrabar      : intrabar model (True) or close-confirmed with next-open fills (False)
    Output:
        df     : OHLCV with U, D, TR, ATR columns
        trades : list of (price_in, price_out)
        dates  : list of (date_in, date_out)
    '''

    df = raw_data[ticker]
    df = data_prep_donchian_breakout(df=df, interval=interval, lookback=lookback, period=period)
    trades, dates = donchian_breakout(df=df, k=k, chandelier=chandelier, intrabar=intrabar)

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


def data_prep_donchian_breakout(
        df: pd.DataFrame,  
        interval: str,
        lookback: int,
        period: int
    ) -> pd.DataFrame:
    '''
    Donchian channel and Wilder ATR for a single ticker.

    Adds columns:
        U   : max High over the previous `lookback` minutes, excluding bar t
        D   : min Low over the previous `lookback` minutes, excluding bar t
        TR  : true range, max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
        ATR : Wilder-smoothed TR, EMA with alpha = 1/period (period in bars);
              NaN for the first 5*period bars (seed burn-in)

    U_t, D_t use data up to t-1 only, so a breakout on bar t is tradable within bar t.
    Bar-count based, so best for tickers without session breaks.
    '''

    lookback_bars = bars_in_period(interval=interval, window=lookback)

    new_df = df.copy()
    ut = new_df['High'].rolling(lookback_bars, closed='left').max()
    dt = new_df['Low'].rolling(lookback_bars, closed='left').min()

    range1 = new_df['High'] - new_df['Low']
    range2 = (new_df['High'] - new_df['Close'].shift(1)).abs()
    range3 = (new_df['Low'] - new_df['Close'].shift(1)).abs()
    true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1, skipna=False)

    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    atr.iloc[:5 * period] = np.nan

    new_df['U'] = ut
    new_df['D'] = dt
    new_df['TR'] = true_range
    new_df['ATR'] = atr

    return new_df
    

def donchian_breakout(
        df: pd.DataFrame,
        k: float, 
        chandelier: bool = False,
        intrabar: bool = False
    ) -> tuple:
    '''
    Long-only Donchian breakout with ATR trailing stop, intrabar or next-open fill model.

    intrabar=True  (intrabar entry):
        Entry : H_t >= U_t, filled at max(O_t, U_t) on bar t; S = entry - k * ATR_{t-1}
        Exit  : L_t <= S_{t-1}, filled at min(O_t, S_{t-1})
    intrabar=False (entry at next bar):
        Entry : C_t > U_t, filled at O_{t+1}; S = entry - k * ATR_t
        Exit  : C_t <= S_{t-1}, filled at O_{t+1}
    The stop is always ratcheted (non-decreasing).

    Returns:
        trades : list of (price_in, price_out)
        dates  : list of (date_in, date_out)
    '''

    in_trade = False
    trades = []
    dates = []
    n = len(df)
    o = df['Open'].to_numpy()
    h = df['High'].to_numpy()
    l = df['Low'].to_numpy()
    c = df['Close'].to_numpy()
    u = df['U'].to_numpy()
    atr = df['ATR'].to_numpy()

    for i in range(n - 1):
        if in_trade:
            s_old = s_new
            if chandelier:
                cur_high = max(cur_high, h[i])
                s_new = cur_high - k * atr[i]
            else:
                s_new = c[i] - k * atr[i]
            
            s_new = max(s_new, s_old)

            breakout = h[i] >= u[i] if intrabar else c[i] > u[i]
            if breakout: # avoid exit + re-entry on same bar
                continue

            if intrabar and l[i] <= s_old:
                out_price, out_i = min(o[i], s_old), i
            elif not intrabar and c[i] <= s_old:
                out_price, out_i = o[i + 1], i + 1
            else:
                continue

            in_trade = False
            trades.append((in_price, out_price))
            dates.append((df.index[entry_i], df.index[out_i]))


        if np.isnan(atr[i - 1]) or np.isnan(u[i]):
            continue
        
        if intrabar:
            if h[i] < u[i]:
                continue
            entry_i = i
            in_price = max(o[i], u[i])
            atr_0 = atr[i - 1]
            cur_high = h[i]
        else:
            if c[i] <= u[i]:
                continue
            entry_i = i + 1
            in_price = o[entry_i]
            atr_0 = atr[i]
            cur_high = in_price

        in_trade = True
        s_new = in_price - k * atr_0
            
    
    return trades, dates

       
# %%

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'lookback': int(60 * 4),
        'period': 14,
        'k': 3, 
        'chandelier': False,
        'intrabar': False
            }
    for ticker in tickers:
        _, _, dates = master_donchian_breakout(ticker=ticker, **hyperparams)
    diffs = np.array([(t2 - t1).total_seconds() / 60 // 5 for t1, t2 in dates])
    median = np.median(diffs) # 51
