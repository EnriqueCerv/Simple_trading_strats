# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.data import raw_data

# %%
def master_basic_momentum(
        ticker: str,
        interval: str, 
        change_period: int,
        in_cond: float, 
        take_profit: float, 
        stop_loss: float, 
        amount: float,
        accumulates: bool = False,
        cost_bps: float = 5.0,
        verbose: bool = True,
        plot: bool = False
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of profits per trade, returns per trade, total profit, cumulative return, optional plot
    '''

    df = raw_data[ticker]
    df = data_prep_basic(df=df, interval=interval, change_period=change_period)
    trades, dates = basic_momentum(df=df, in_cond=in_cond, take_profit=take_profit, stop_loss=stop_loss)
    profits, returns, total_profit = eval_strat(trades=trades, amount=amount, accumulates=accumulates, cost_bps=cost_bps)

    returns = np.array(returns)
    cum_return = (1 + returns).cumprod() if len(returns) else np.array([1.0])

    win_pct = sum(1 for ele in profits if ele > 0) / len(trades) if trades else 0
    n_trades = len(trades)

    if verbose:        
        print(f'Ticker: {ticker}')
        print(f'Number of trades in {df.index.normalize().nunique()} days: {n_trades}')
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
def data_prep_basic(
        df: pd.DataFrame,  
        interval: str, 
        change_period: int
    ) -> pd.DataFrame:
    '''
    Input: DataFrame of a single ticker, ticker frequency, lookback window
    Output: Same dataframe with lagged returns over last (change_period / interval) tickers 
    (Best suited for tickers with no session breaks as lag is ticker based not time based)    
    '''

    interval = int(interval[:-1])
    n_bars = change_period // interval

    new_df = df.copy()
    new_df['pct_change'] = new_df['Close'].pct_change(n_bars)

    return new_df


def basic_momentum(
        df: pd.DataFrame, 
        in_cond: float, 
        take_profit: float, 
        stop_loss: float,
    ) -> tuple:
    '''
    Input: Dataframe, momentum condition to enter a trade, take_profit and stop_loss conditions
    Output: A tuple of (price_in, price_out) for each trade entered, respective returns and dates (date_in, date_out)
    '''

    in_trade = False
    trades = []
    dates = []

    for idx in df.dropna(subset=['pct_change']).index:
        cur_price = df['Close'].loc[idx]
        if not in_trade:
            if df['pct_change'].loc[idx] >= in_cond:
                in_trade = True
                in_price = cur_price
                in_date = idx

        else:
            up_cond = cur_price > in_price * take_profit
            down_cond = cur_price < in_price * stop_loss

            if up_cond or down_cond:
                in_trade = False
                out_date = idx
                trades.append((in_price, cur_price))
                dates.append((in_date, out_date))
    
    return trades, dates

def eval_strat(
        trades: list,
        amount: float,
        accumulates: bool = False,
        cost_bps: float = 5.0
    ) -> tuple:
    '''
    Input: tuples of trades from basic_momentum, amount to buy_in, boolean that determines whether we reinvest
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
        'in_cond': 0.01, 
        'take_profit': 1.01, 
        'stop_loss': 0.99, 
        'amount': 10000,
        'accumulates': False,
        'cost_bps': 5.0,
        'verbose': True,
        'plot': True
            }
    for ticker in tickers:
        _,_,_,_ = master_basic_momentum(ticker = ticker, **hyperparams)
