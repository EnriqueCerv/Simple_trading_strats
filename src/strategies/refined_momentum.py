# %%
import pandas as pd
import numpy as np

# %%
def master_refined_momentum(
        ticker: str,
        data: dict,
        interval: str, 
        change_period: int,
        in_cond: float, 
        take_profit: float, 
        stop_loss: float
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of dataframe, trade tuples (price_in, price_out) and respective dates
    '''

    df = data[ticker]
    df = data_prep_refined(df=df, interval=interval, change_period=change_period)
    trades, dates = refined_momentum(df=df, in_cond=in_cond, take_profit=take_profit, stop_loss=stop_loss)

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


def data_prep_refined(
        df: pd.DataFrame,  
        interval: str, 
        change_period: int
    ) -> pd.DataFrame:
    '''
    Input: DataFrame of a single ticker, ticker frequency, lookback window
    Output: Same dataframe with lagged returns over last (change_period / interval) tickers 
    (Best suited for tickers with no session breaks as lag is ticker based not time based)    
    '''

    n_bars = bars_in_period(interval=interval, window=change_period)

    new_df = df.copy()
    new_df['pct_change'] = new_df['Close'].pct_change(n_bars)

    return new_df


def refined_momentum(
        df: pd.DataFrame, 
        in_cond: float, 
        take_profit: float, 
        stop_loss: float,
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
    c = df['Close'].to_numpy()
    pct = df['pct_change'].to_numpy()

    for i in range(n):
        if not in_trade:
            if np.isnan(pct[i]) or i + 1 == n or pct[i] < in_cond:
                continue
            
            in_trade = True
            entry_i = i + 1
            in_price = o[entry_i]

        else:
            cur_close = c[i]
            up_cond = cur_close > in_price * take_profit
            down_cond = cur_close < in_price * stop_loss

            if up_cond or down_cond:
                if i + 1 == n:
                    break
                in_trade = False
                exit_i = i + 1
                trades.append((in_price, o[exit_i]))
                dates.append((df.index[entry_i], df.index[exit_i]))
    
    return trades, dates

# %%

if __name__ == '__main__':

    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']

    from src.data import get_data_yf
    raw_data = get_data_yf('5m')

    hyperparams = {
        'interval': '5m',
        'data': raw_data, 
        'change_period': 240,
        'in_cond': 0.01, 
        'take_profit': 1.01, 
        'stop_loss': 0.99
            }
    for ticker in tickers:
        master_refined_momentum(ticker = ticker, **hyperparams)
