# %%
import os
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
# %%
def load(ticker, interval, period, start=None, end=None, plot=True):
    df = yf.Ticker(ticker=ticker)
    if start is not None:
        df = df.history(start=start, end=end, interval=interval)
    else:
        df = df.history(period=period, interval=interval)

    df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

    if plot:
        mpf.plot(df, type='candle', figsize=(12,6), style="yahoo", title=f"{ticker}")

    return df

def get_data_yf(interval):
    cur_path = Path(__file__).resolve()
    project_root = next(p for p in cur_path.parents if p.name == "Simple_trading_strats")
    data_dir = os.path.join(os.path.join(project_root, 'Data'), 'yf')
    data_csv = os.path.join(data_dir, f'raw_data_{interval}.csv')

    combined_df = pd.read_csv(data_csv, index_col=[0, 1], parse_dates=True)
    raw_data = {ticker: combined_df.xs(ticker, level='Ticker') for ticker in tickers if ticker in combined_df.index.levels[0]}

    return raw_data
# %%
tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDT-USD', 'USDC-USD', 'XRP-USD']
intervals = ['5m', '30m', '60m', '1d']
intervals = ['1d']
period = 'max'

# project_root = os.path.abspath(os.path.dirname(__file__))
cur_path = Path(__file__).resolve()
project_root = next(p for p in cur_path.parents if p.name == "Simple_trading_strats")
data_dir = os.path.join(os.path.join(project_root, 'Data'), 'yf')
os.makedirs(data_dir, exist_ok=True)

for interval in intervals:
    name = f'raw_data_{interval}.csv'
    data_csv = os.path.join(data_dir, name)

    if os.path.exists(data_csv):
        print(f'{interval} Data already saved as csv')
        # print("Loading existing data from CSV...")
        # combined_df = pd.read_csv(data_csv, index_col=[0, 1], parse_dates=True)
        
        # raw_data = {ticker: combined_df.xs(ticker, level='Ticker') for ticker in tickers if ticker in combined_df.index.levels[0]}
    else:
        print("Fetching new data from yfinance...")
        raw_data = {ticker: load(ticker, interval=interval, period=period, plot=False) for ticker in tickers}
        
        combined_df = pd.concat(raw_data.values(), keys=raw_data.keys(), names=['Ticker', 'Date'])
        
        combined_df.to_csv(data_csv)
        print(f"{interval} Data successfully saved to {data_csv}")

# %%
if __name__ == '__main__':
    for ticker in tickers:
        interval = '5m'
        period = 'max'
        load(ticker, interval=interval, period=period)