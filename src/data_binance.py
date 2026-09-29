# %%
import os
import io
import zipfile
import hashlib
from datetime import date
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import mplfinance as mpf

# %%
BASE_URL = 'https://data.binance.vision/data/spot'

KLINE_COLS = ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'close_time', 'quote_volume', 'n_trades',
              'taker_buy_base', 'taker_buy_quote', 'ignore']
KEEP_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# yfinance-style keys -> Binance symbols, so downstream code keeps raw_data['BTC-USD']
# (USDT-USD has no Binance equivalent: USDT is the quote asset)
SYMBOLS = {
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT',
    'LTC-USD': 'LTCUSDT',
    'XRP-USD': 'XRPUSDT',
    'USDC-USD': 'USDCUSDT',
}


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    unit = {'m': 'min', 'h': 'h', 'd': 'D'}[interval[-1].lower()]
    return pd.Timedelta(int(interval[:-1]), unit=unit)


def _monthly_url(symbol, interval, period):
    return (f'{BASE_URL}/monthly/klines/{symbol}/{interval}/'
            f'{symbol}-{interval}-{period.year}-{period.month:02d}.zip')


def _daily_url(symbol, interval, day):
    return (f'{BASE_URL}/daily/klines/{symbol}/{interval}/'
            f'{symbol}-{interval}-{day:%Y-%m-%d}.zip')


def _fetch(url: str, cache_dir: str, remember_404: bool = False, verify: bool = True):
    '''
    Download one zip (cached on disk). Returns bytes, or None if not published.
    remember_404: cache the miss (for months before listing) so it isn't re-requested.
    '''
    fname = url.rsplit('/', 1)[-1]
    path = os.path.join(cache_dir, fname)
    miss_marker = path + '.404'

    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    if os.path.exists(miss_marker):
        return None

    os.makedirs(cache_dir, exist_ok=True)
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        if remember_404:
            open(miss_marker, 'w').close()
        return None
    r.raise_for_status()
    content = r.content

    if verify:
        chk = requests.get(url + '.CHECKSUM', timeout=60)
        if chk.status_code == 200:
            expected = chk.text.split()[0]
            actual = hashlib.sha256(content).hexdigest()
            if expected != actual:
                raise ValueError(f'checksum mismatch for {fname}')

    with open(path, 'wb') as f:
        f.write(content)
    return content


def _parse(content: bytes) -> pd.DataFrame:
    '''Parse one kline zip into an OHLCV frame indexed by UTC open time.'''
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLS)

    if not str(df.iloc[0, 0]).isdigit():   # tolerate files that ship a header row
        df = df.iloc[1:]

    ts = df['open_time'].astype('int64')
    unit = 'us' if ts.iloc[0] > 1e14 else 'ms'   # spot switched ms -> us on 2025-01-01

    out = df[KEEP_COLS].astype(float)
    out.index = pd.to_datetime(ts.to_numpy(), unit=unit, utc=True)
    out.index.name = 'Date'
    return out


def load(ticker: str, interval: str, start: str, end: str | None = None,
         cache_root: str = 'Data/binance', fill_gaps: bool = False,
         n_workers: int = 8) -> pd.DataFrame:
    '''
    Spot klines from data.binance.vision for one ticker.

    Input:
        ticker    : yfinance-style key ('BTC-USD') or raw Binance symbol ('BTCUSDT')
        interval  : '1m', '5m', '1h', '1d', ...
        start/end : dates 'YYYY-MM-DD'; end=None means up to yesterday (UTC)
        fill_gaps : reindex to the full bar grid; missing bars get O=H=L=C=previous close, V=0
    Output:
        OHLCV DataFrame, UTC DatetimeIndex named 'Date'

    Complete months come from monthly archives. The last ~2 months fall back to
    daily archives when the monthly file is not published yet.
    '''
    symbol = SYMBOLS.get(ticker, ticker)
    cache_dir = os.path.join(cache_root, symbol, interval)
    today = pd.Timestamp.now(tz='UTC').date()
    end_date = pd.Timestamp(end).date() if end else today

    months = pd.period_range(start=start, end=end_date, freq='M')
    cur = today.year * 12 + today.month
    is_recent = [cur - (p.year * 12 + p.month) <= 1 for p in months]

    with ThreadPoolExecutor(n_workers) as ex:
        monthly = list(ex.map(
            lambda args: _fetch(_monthly_url(symbol, interval, args[0]), cache_dir,
                                remember_404=not args[1]),
            zip(months, is_recent)))
    frames = [_parse(c) for c in monthly if c is not None]

    # recent months without a monthly archive -> daily files
    days = [d.date()
            for p, c, rec in zip(months, monthly, is_recent) if c is None and rec
            for d in pd.date_range(p.start_time, p.end_time, freq='D')
            if d.date() < today]
    with ThreadPoolExecutor(n_workers) as ex:
        daily = list(ex.map(lambda d: _fetch(_daily_url(symbol, interval, d), cache_dir), days))
    frames += [_parse(c) for c in daily if c is not None]

    if not frames:
        raise ValueError(f'no data found for {symbol} {interval} from {start}')

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]

    t0 = pd.Timestamp(start, tz='UTC')
    t1 = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    df = df[(df.index >= t0) & (df.index < t1)]

    step = _interval_to_timedelta(interval)
    full = pd.date_range(df.index[0], df.index[-1], freq=step)
    missing = full.difference(df.index)
    if len(missing):
        print(f'{ticker}: {len(missing)} missing bars ({len(missing) / len(full):.3%})')

    if fill_gaps and len(missing):
        df = df.reindex(full)
        df['Volume'] = df['Volume'].fillna(0.0)
        df['Close'] = df['Close'].ffill()
        for col in ['Open', 'High', 'Low']:
            df[col] = df[col].fillna(df['Close'])
        df.index.name = 'Date'

    return df


# %%
tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']
interval = '5m'
start = '2018-01-01'
end = None          # up to yesterday
refresh = False     # True: rebuild the combined file (zips stay cached, so it's fast)

from pathlib import Path

def _find_src() -> Path:
    try:
        here = Path(__file__).resolve().parent
    except NameError:                  # interactive window / notebook
        here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if p.name == 'src':
            return p
    raise FileNotFoundError(f'could not locate src/ above {here}')

data_dir = _find_src() / 'Data'
cache_root = data_dir / 'binance'
data_dir.mkdir(parents=True, exist_ok=True)
data_file = data_dir / f'raw_data_binance_{interval}.parquet'

if os.path.exists(data_file) and not refresh:
    print('Loading existing data from parquet...')
    combined_df = pd.read_parquet(data_file)
    raw_data = {t: combined_df.xs(t, level='Ticker')
                for t in tickers if t in combined_df.index.get_level_values('Ticker')}
else:
    print('Fetching data from data.binance.vision...')
    raw_data = {t: load(t, interval=interval, start=start, end=end, cache_root=cache_root)
                for t in tickers}
    combined_df = pd.concat(raw_data.values(), keys=raw_data.keys(), names=['Ticker', 'Date'])
    combined_df.to_parquet(data_file)
    print(f'Data saved to {data_file}')

# %%
if __name__ == '__main__':
    for t, df in raw_data.items():
        print(f'{t}: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}')
        mpf.plot(df.iloc[-576:], type='candle', figsize=(12, 6), style='yahoo',
                 title=f'{t} (last 2 days)')