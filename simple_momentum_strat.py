# %%
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
# %%
def load(ticker, interval, period, start=None, end=None, plot=True):
    df = yf.Ticker(ticker=ticker)
    if start is not None:
        df = df.history(start=start, end=end, interval=interval)
    else:
        df = df.history(period=period, interval=interval)

    if plot:
        mpf.plot(df, type='candle', figsize=(12,6), style="yahoo", title=f"{ticker}")

    return df
    # return df['Close']

interval = '5m'
period = 'max'
aapl = load('AAPL', period=period, interval=interval)

# %%
def data_prep(df, interval, change_period):
    interval = int(interval[:-1])
    n_bars = change_period // interval

    # df[f'{change_period}h_pct_change'] = df.groupby(df.index.date)['Close'].pct_change(n_bars)
    df['pct_change'] = df['Close'].pct_change(n_bars)

    return df

change_period = 240
aapl = data_prep(aapl, interval=interval, change_period=change_period)
aapl
# %%
def basic_strat(df, in_cond, out_up_cond, out_down_cond, amount, cost_bps=0):
    
    in_trade = False
    trades = []
    dates = []
    gain = 0
    n_trades = 0

    for idx in df.dropna(subset=['pct_change']).index:
        cur_price = df['Close'].loc[idx]
        if not in_trade:
            if df['pct_change'].loc[idx] >= in_cond:
                in_trade = True
                in_price = cur_price
                in_date = idx
                n_stocks = amount / cur_price

        else:
            up_cond = cur_price > in_price * out_up_cond
            down_cond = cur_price < in_price * out_down_cond

            if up_cond or down_cond:
                in_trade = False
                out_date = idx
                gross = (cur_price - in_price) * n_stocks
                cost  = (in_price + cur_price) * n_stocks * cost_bps / 1e4
                trades.append(gross - cost)
                dates.append((in_date, out_date))
                gain += gross - cost
                # amount += gross - cost
                n_trades += 1
    
    return trades, gain, n_trades, dates

trades, gain, n_trades, _ = basic_strat(
    aapl, 
    in_cond=0.04, 
    out_up_cond=1.01,
    out_down_cond=0.99,
    amount=10000
    )
# gain, n_trades, trades
# %%
def master(
        ticker, interval='5m', period='max', start=None, end=None, plot=False,
        change_period=240, 
        in_cond=0.01, out_up_cond=1.01, out_down_cond=0.99, amount=10000, cost_bps=5.0
        ):
    
    df = load(ticker=ticker, interval=interval, period=period, start=start, end=end, plot=False)
    df = data_prep(df=df, interval=interval, change_period=change_period)
    trades, gain, n_trades, dates = basic_strat(df=df, in_cond=in_cond, out_up_cond=out_up_cond, out_down_cond=out_down_cond, amount=amount, cost_bps=cost_bps)

    win_pct = sum(1 for ele in trades if ele > 0) / len(trades) if trades else 0

    print(f'Ticker: {ticker}')
    print(f'Number of trades in {df.index.normalize().nunique()} days: {n_trades}')
    print(f'Total profit with {amount}USD per trade: {gain}')
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
            figsize=(16, 4),
            style='yahoo',
            title=f'{ticker}  ({n_trades} trades, {gain:+.0f} USD)',
            addplot=aps if aps else None,
            warn_too_much_data=len(df) + 1,
            tight_layout=True,
        )


    return trades, gain, n_trades, dates


trades, gain, n_trades, dates = master('BTC-USD', start='2026-09-14', end='2026-09-21', plot=True)

# %%

tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD']
# tickers = ['BTC', 'ETH', 'LTC']
for ticker in tickers:
    _,_,_,_ = master(ticker, out_up_cond = 1.02, cost_bps=5, plot= True)

