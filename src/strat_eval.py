# %%
import pandas as pd
import numpy as np
import mplfinance as mpf

# %%
def strat_data(
        ticker: str,
        df: pd.DataFrame,
        trades: list, 
        dates: list,
        amount: float,
        strategy: str = None, 
        n_ambiguous: int = None,
        accumulates: bool = False,
        cost_bps: float = 5.0,
        verbose: bool = True,
        plot: bool = False
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy
    Output: Tuple of profits per trade, returns per trade, total profit, cumulative return, optional plot
    '''

    profits, returns, total_profit = eval_strat(trades=trades, amount=amount, accumulates=accumulates, cost_bps=cost_bps)

    returns = np.array(returns)
    cum_return = (1 + returns).cumprod() if len(returns) else np.array([1.0])

    win_pct = sum(1 for ele in profits if ele > 0) / len(trades) if trades else 0
    n_trades = len(trades)

    if verbose:
        print(f"Ticker: {ticker}{f'  |  Strategy: {strategy}' if strategy else ''}")
        print(f'Number of trades in {df.index.normalize().nunique()} days: {n_trades}')
        if n_trades and n_ambiguous:
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
            title=f'{ticker}{f" — {strategy}" if strategy else ""}  '
                  f'({n_trades} trades, {float(total_profit):+.0f} USD)',
            addplot=aps if aps else None,
            warn_too_much_data=len(df) + 1,
            tight_layout=True,
        )


    return profits, total_profit, cum_return, cum_return[-1]


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

    from src.strategies.basic_momentum import master_basic_momentum
    from src.strategies.refined_momentum import master_refined_momentum
    from src.strategies.accurate_momentum import master_accurate_momentum
    from src.strategies.vol_adjusted_momentum import master_vol_adjusted_momentum
    from src.strategies.momentum_fixed_horizon import master_momentum_fixed_horizon

    ticker = 'BTC-USD'

    barrier_params = {
        'interval': '5m',
        'change_period': 240,
        'in_cond': 0.01,
        'take_profit': 1.015,
        'stop_loss': 0.985,
    }

    strat_params = {
        'basic':        {'fn': master_basic_momentum,   'params': barrier_params},
        'refined':      {'fn': master_refined_momentum, 'params': barrier_params},
        'accurate':     {'fn': master_accurate_momentum,'params': barrier_params},
        'vol_adjusted': {'fn': master_vol_adjusted_momentum, 'params': {
            'interval': '5m',
            'change_period': 240,
            'z_in': 1.8,
            'tp_sigma': 1.6,
            'sl_sigma': 1.6,
            'vol_window': 576,
            'tau_mult': 3.0,
            'min_sigma': 0.0,
        }},
        'fixed_horizon': {'fn': master_momentum_fixed_horizon, 'params': {
            'interval': '5m',
            'change_period': 240,
            'z_in': 1.8,
            'horizon': 22
        }}
    }

    base_eval = {
        'amount': 10000,
        'accumulates': True,
        'cost_bps': 10.0,
        'verbose': True,
        'plot': True,
    }
    eval_params = {name: dict(base_eval) for name in strat_params}

    results = {}
    for name, cfg in strat_params.items():
        out = cfg['fn'](ticker=ticker, **cfg['params'])

        # basic/refined return 3 values, accurate/vol_adjusted return 4
        df, trades, dates = out[:3]
        n_ambiguous = out[3] if len(out) > 3 else None

        results[name] = strat_data(
            ticker, df, trades, dates,
            strategy=name,
            n_ambiguous=n_ambiguous,
            **eval_params[name]
        )

    print(f'{"strategy":<14}{"trades":>8}{"win_pct":>10}{"profit":>14}{"cum_return":>12}')
    for name, (profits, total_profit, cum_return, final_return) in results.items():
        win_pct = sum(1 for p in profits if p > 0) / len(profits) if profits else 0.0
        print(f'{name:<14}{len(profits):>8}{win_pct:>10.1%}'
              f'{float(total_profit):>14,.0f}{float(final_return):>12.3f}')

# %%
