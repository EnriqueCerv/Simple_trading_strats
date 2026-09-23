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
        amount: float,
        accumulates: bool = False,
        cost_bps: float = 5.0,
        vol_window: int = 576,
        tau_mult: float = 3.0,
        min_sigma: float = 0.0,
        verbose: bool = True,
        plot: bool = False
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of profits per trade, returns per trade, total profit, cumulative return, optional plot
    '''

    df = raw_data[ticker]
    df, n_bars = data_prep_vol(df=df, interval=interval, change_period=change_period, vol_window=vol_window)
    trades, dates, n_ambiguous = vol_adjusted_momentum(df=df, z_ind=z_ind, tp_sigma=tp_sigma, sl_sigma=sl_sigma,
                                                       n_bars=n_bars, tau_mult=tau_mult, min_sigma=min_sigma)
    profits, returns, total_profit = eval_strat(trades=trades, amount=amount, accumulates=accumulates, cost_bps=cost_bps)

    returns = np.array(returns)
    cum_return = (1 + returns).cumprod() if len(returns) else np.array([1.0])

    win_pct = sum(1 for ele in profits if ele > 0) / len(trades) if trades else 0
    n_trades = len(trades)

    if verbose:        
        print(f'Ticker: {ticker}')
        print(f'Number of trades in {df.index.normalize().nunique()} days: {n_trades}')
        if n_trades:
            print(f'Ambiguous exits: {n_ambiguous} ({n_ambiguous / n_trades:.1%})')
        print(f"Total profit starting with {amount}USD, ({'accumulating' if accumulates else 'constant'}): {total_profit}")
        print(f'Cumulative return: {cum_return[-1]}')
        print(f'Win percentage: {win_pct:.3f}')
        print()

    if plot:
        buy_marks  = pd.Series(np.nan, index=df.index)
        sell_marks = pd.Series(np.nan, index=df.index)

        for buy_time, sell_time in dates:
            if buy_time in df.index:
                buy_marks.at[buy_time] = df.at[buy_time, 'Low'] * 0.998
            if sell_time is not None and sell_time in df.index:
                sell_marks.at[sell_time] = df.at[sell_time, 'High'] * 1.002

        aps = []
        if buy_marks.notna().any():
            aps.append(mpf.make_addplot(buy_marks, type='scatter',
                                        marker='^', markersize=90, color='lime'))
        if sell_marks.notna().any():
            aps.append(mpf.make_addplot(sell_marks, type='scatter',
                                        marker='v', markersize=90, color='red'))

        mpf.plot(
            df,
            type='candle',
            figsize=(16, 6),
            style='yahoo',
            title=f'{ticker}  ({n_trades} trades, {total_profit:+.0f} USD)',
            addplot=aps if aps else None,
            warn_too_much_data=len(df) + 1,
            tight_layout=True,
        )


    return profits, total_profit, cum_return, cum_return[-1]
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

def eval_strat(
        trades: list,
        amount: float,
        accumulates: bool = False,
        cost_bps: float = 5.0
    ) -> tuple:
    '''
    Input: tuples of trades from the strategy loop, amount to buy_in, boolean that determines whether we reinvest
    Output: net_profit and return per trade, total net_profit
    '''

    profits = []
    returns = []
    total_profit = 0

    for in_price, out_price in trades:
        n_stocks = amount / in_price
        gross = (out_price - in_price) * n_stocks
        cost  = (in_price + out_price) * n_stocks * cost_bps / 1e4

        net_profit = gross - cost
        profits.append(net_profit)
        returns.append(net_profit / amount)
        total_profit += net_profit
        amount += net_profit if accumulates else 0

    return profits, returns, total_profit
       
# %%

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'change_period': 240,
        'z_ind': 1.75, 
        'tp_sigma': 1.4, 
        'sl_sigma': 1.4, 
        'amount': 10000,
        'accumulates': True,
        'cost_bps': 10.0,
        'vol_window': 576,
        'tau_mult': 3.0,
        'min_sigma': 0.0,
        'verbose': True,
        'plot': True
            }
    for ticker in tickers:
        master_vol_adjusted_momentum(ticker=ticker, **hyperparams)
