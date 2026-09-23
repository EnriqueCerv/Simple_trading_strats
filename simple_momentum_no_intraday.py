# %%
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
# %%
def load(ticker, interval, period, start=None, end=None, plot=True):
    df = yf.Ticker(ticker=ticker)
    if start is not None:
        df = df.history(start=start, end=end, interval=interval)
    else:
        df = df.history(period=period, interval=interval)

    if plot:
        mpf.plot(df, type='candle', style="yahoo", title=f"{ticker}")

    return df
    # return df['Close']

interval = '5m'
period = 'max'
aapl = load('AAPL', period=period, interval=interval)

# %%
def data_prep(df, interval, change_period):
    interval = int(interval[:-1])
    n_bars = change_period // interval

    df['pct_change'] = df.groupby(df.index.date)['Close'].pct_change(n_bars)
    df['is_last'] = df.index.isin(df.groupby(df.index.date).tail(1).index)

    return df

change_period = 60
aapl = data_prep(aapl, interval=interval, change_period=change_period)
aapl


# %%
def basic_strat(df, in_cond, out_up_cond, out_down_cond, amount, cost_bps=0.0):
    open = df['Open'].to_numpy()
    close = df['Close'].to_numpy()
    pct = df['pct_change'].to_numpy()
    last = df['is_last'].to_numpy()
    n = len(df)

    in_trade = False
    trades = []
    gain = 0
    n_trades = 0
    forced = 0

    for i in range(n):
        if not in_trade:
            if np.isnan(pct[i]) or last[i] or pct[i] < in_cond or i + 1 == n:
                continue

            entry_bar = i + 1
            in_price = open[entry_bar]
            n_stocks = amount / in_price
            in_trade = True

        else:
            if last[i]:
                in_trade = False
                out_price = open[i]
                trades.append((out_price - in_price) * n_stocks)
                gain += (out_price - in_price) * n_stocks
                n_trades += 1
                forced += 1

            else:
                cur_price = close[i]
                up_cond = cur_price > in_price * out_up_cond
                down_cond = cur_price < in_price * out_down_cond

                if up_cond or down_cond:
                    in_trade = False
                    out_price = open[i + 1]
                    trades.append((out_price - in_price) * n_stocks)
                    gain += (out_price - in_price) * n_stocks
                    n_trades += 1
    
    return trades, gain, n_trades, forced


trades, gain, n_trades, forced = basic_strat(
    aapl,
    in_cond=0.01, 
    out_up_cond=1.01,
    out_down_cond=0.99,
    amount=10000
    )
gain, n_trades, trades, forced
# %%
def master(
        ticker, interval='5m', period='max', 
        change_period=60, 
        in_cond=0.01, out_up_cond=1.01, out_down_cond=0.99, amount=10000
        ):
    
    df = load(ticker=ticker, interval=interval, period=period, plot=False)
    df = data_prep(df=df, interval=interval, change_period=change_period)
    trades, gain, n_trades, forced = basic_strat(df=df, in_cond=in_cond, out_up_cond=out_up_cond, out_down_cond=out_down_cond, amount=amount)

    win_pct = sum(1 for ele in trades if ele > 0) / len(trades) if trades else 0

    print(f'Ticker: {ticker}')
    print(f'Number of trades in last 60 days: {n_trades} (of which forced: {forced})')
    print(f'Total profit with {amount}USD per trade: {gain}')
    print(f'Win percentage: {win_pct:.3f}')
    print()

    return trades, gain, n_trades, forced

tickers = ['VOO', 'SMH', 'AAPL', 'TSM', 'NVDA', 'GLD', 'TSLA', 'MSFT']
trades, gain, n_trades, forced = master('AAPL')