# %%
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from typing import Callable

from src.strategies.accurate_momentum import master_accurate_momentum
from src.strategies.vol_adjusted_momentum import master_vol_adjusted_momentum
from src.strategies.trend_reversal import master_trend_reversal
from src.strategies.donchian_breakout import master_donchian_breakout

from src.grid_search import get_optimal_params
from src.strat_eval import eval_strat, restrict_to_window
# %%


def backtest_returns(
        ticker: str,
        data: dict,
        interval: str,
        strategy: str,
        lookback: int,
        rebalance_freq: int,
        eval_kwargs: dict,
        rank_by: str,
        min_trades: int,
        verbose: bool = True,
    ):
    
    amount = eval_kwargs['amount']
    accumulates = eval_kwargs.get('accumulates', True)
    cost_bps = eval_kwargs.get('cost_bps', 10.0)

    rolling_trades, rolling_dates, params_log = compute_rolling_params(
        ticker=ticker,
        data=data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_kwargs,
        rank_by=rank_by,
        min_trades=min_trades,
        verbose=verbose,
    )

    rolling_pnl, rolling_returns, _ = rolling_eval_strat(
        rolling_trades=rolling_trades,
        amount=amount,
        accumulates=accumulates,
        cost_bps=cost_bps
    )

    rolling_returns = np.asarray(rolling_returns, dtype=float)
    cum_rolling_returns = ((1 + rolling_returns).cumprod()
                           if len(rolling_returns) else np.array([]))
    rolling_profit = amount * (cum_rolling_returns[-1] - 1) if len(cum_rolling_returns) else 0.0

    exit_dates = pd.DatetimeIndex([d_out for _, d_out in rolling_dates])
    rolling_returns = pd.Series(rolling_returns, index=exit_dates)
    cum_rolling_returns = pd.Series(cum_rolling_returns, index=exit_dates)

    return rolling_pnl, rolling_profit, rolling_returns, cum_rolling_returns, rolling_dates, params_log
# %%

# # # # # # # # # # # # #
# Rolling portfolio optimizer
# # # # # # # # # # # # #

def compute_rolling_params(
        ticker: str,
        data: dict,
        interval: str,
        strategy: str,
        eval_kwargs: dict,
        rank_by: str,
        min_trades: int,
        lookback: int,
        rebalance_freq: int,
        verbose: bool = True,
    ):
    '''
    For each rebalance time t:
      train = [t - lookback, t)          -> grid search picks params
      test  = [t, t + rebalance_freq)    -> trade with those params
    The test run is fed [t - lookback, t + rebalance_freq) so indicators are warm
    at t; only trades ENTERED in [t, t_end) are kept, and any still open at t_end
    are marked to the last close before t_end. Windows are therefore disjoint.
    '''
    spec = strat_params[strategy]
    fn, base_params = spec['fn'], spec['params']
    df = data[ticker]

    rolling_trades, rolling_dates, params_log = [], [], []

    starts = list(range(lookback, len(df), rebalance_freq))
    for i, t in enumerate(starts, 1):
        t_start = df.index[t]
        t_stop = min(t + rebalance_freq, len(df))
        t_end = df.index[t_stop] if t_stop < len(df) else None    # exclusive bound
        if verbose:
            print(f'[{i}/{len(starts)}] {t_start:%Y-%m-%d} for {strategy}', flush=True)

        train = df.iloc[t - lookback : t].dropna()
        if len(train) < lookback:
            continue

        optimal_params = get_optimal_params(
            strategy=strategy,
            ticker=ticker,
            data={ticker: train},
            grid=param_grids[interval][strategy],
            eval_kwargs=eval_kwargs,
            rank_by=rank_by,
            min_trades=min_trades,
            constraint=constraints.get(strategy)
        )
        if optimal_params is None:
            if verbose:
                print('    no eligible params -> flat this period')
            params_log.append({'start': t_start})
            continue
        params_log.append({'start': t_start, **optimal_params})

        full_optimal_params = base_params | optimal_params | {'interval': interval}

        test = df.iloc[t - lookback : t_stop]
        out = fn(ticker=ticker, data={ticker: test}, **full_optimal_params)
        test_df, trades, dates = out[:3]

        _, trades, dates = restrict_to_window(test_df, trades, dates, start=t_start, end=t_end)
        if not trades:
            continue

        rolling_trades.append(trades)
        rolling_dates += dates

    params_log = pd.DataFrame(params_log).set_index('start') if params_log else pd.DataFrame()
    return rolling_trades, rolling_dates, params_log


def rolling_eval_strat(
        rolling_trades: list,
        amount: float = 0,
        accumulates: bool = False,
        cost_bps: float = 10.0
    ) -> tuple:

    rolling_pnl = []
    rolling_returns = []
    rolling_profit = 0

    for trades in rolling_trades:
        pnl, returns, profit = eval_strat(
            trades=trades,
            amount=amount,
            accumulates=accumulates,
            cost_bps=cost_bps
            )
        
        rolling_pnl += pnl
        rolling_returns += returns
        rolling_profit += profit

    return rolling_pnl, rolling_returns, rolling_profit
# %%

# ---------------------------------------------------------------------------
# Base parameters per strategy (grid values override these) for 5min intervals
# ---------------------------------------------------------------------------
barrier_params = {
    'interval': '5m',
    'change_period': 240,
    'in_cond': 0.01,
    'take_profit': 1.015,
    'stop_loss': 0.985,
}

strat_params = {
    'accurate':     {'fn': master_accurate_momentum, 'params': barrier_params},
    'vol_adjusted': {'fn': master_vol_adjusted_momentum, 'params': {
        'interval': '5m',
        'change_period': 240,
        'z_in': 1.8,
        'tp_sigma': 1.6,
        'sl_sigma': 1.6,
        'vol_window': 576*5,
        'tau_mult': 3.0,
        'min_sigma': 0.0,
    }},
    'trend_reversal': {'fn': master_trend_reversal, 'params': {
        'interval': '5m',
        'fast_window': 60,
        'slow_window': 180,
        'in_cond': 1.0,
        'out_cond': -0.5,
        'burn_spans': 3,
    }},
    'donchian_breakout': {'fn': master_donchian_breakout, 'params':{
        'interval': '5m', 
        'lookback': int(60 * 4),
        'period': 14*5,
        'k': 3, 
        'chandelier': False, 
        'intrabar': True
    }}
}

# ---------------------------------------------------------------------------
# Grids to search per strategy 
# ---------------------------------------------------------------------------
param_grids = {
    'basic':         {'in_cond': [0.005, 0.01, 0.02],
                      'take_profit': [1.01, 1.015, 1.02],
                      'stop_loss': [0.98, 0.985, 0.99],
                      'change_period': [60, 120, 180, 240, 300]},
    'refined':       {'in_cond': [0.005, 0.01, 0.02],
                      'take_profit': [1.01, 1.015, 1.02],
                      'stop_loss': [0.98, 0.985, 0.99],
                      'change_period': [60, 120, 180, 240, 300]},
    'accurate':      {'in_cond': [0.005, 0.01, 0.02],
                      'take_profit': [1.01, 1.015, 1.02],
                      'stop_loss': [0.98, 0.985, 0.99],
                      'change_period': [60, 120, 180, 240, 300]},
    'vol_adjusted':  {'z_in': [1.5, 1.8, 2.1],
                      'tp_sigma': [1.2, 1.6, 2.0],
                      'sl_sigma': [1.2, 1.6, 2.0],
                      'change_period': [60, 120, 180, 240, 300]},
    'fixed_horizon': {'z_in': [1.5, 1.8, 2.1],
                      'horizon': [12*5, 22*5, 48*5]},
    'trend_reversal': {'in_cond': [0.0, 0.5, 1.0, 1.5, 2.0],
                       'out_cond': [-1.0, -0.5, 0.0]},
    'short_reversal': {'z_in': [1.5, 2, 2.5, 3, 3.5],
                       'k_halflife': [1, 1.5, 2, 2.5, 3],
                       },
    'donchian_breakout': { # for 5min bars
                        'lookback':   [240, 720, 1440, 2880, 10080],  # minutes: 4h, 12h, 1d, 2d, 1w
                        'period':     [14*5, 48*5, 288*5],                  # 5m bars: 70 min, 4h, 1d
                        'k':          [2, 3, 5, 8, 12],               # 5m ATR, so larger k for longer lookbacks
                        'chandelier': [False, True],
                        'intrabar': [False, True]
                        }
}

param_grids = {
    '5m':{
        'accurate':      {'in_cond': [0.005, 0.01, 0.02],
                              'take_profit': [1.01, 1.015, 1.02],
                              'stop_loss': [0.98, 0.985, 0.99],
                              'change_period': [60, 120, 180, 240, 300]},
        'vol_adjusted':  {'z_in': [1.5, 1.8, 2.1],
                            'tp_sigma': [1.2, 1.6, 2.0],
                            'sl_sigma': [1.2, 1.6, 2.0],
                            'change_period': [60, 120, 180, 240, 300]},
        'trend_reversal': {'in_cond': [0.0, 0.5, 1.0, 1.5, 2.0],
                            'out_cond': [-1.0, -0.5, 0.0],
                            'fast_window': [30, 60, 120, 240, 720],      # 30m, 1h, 2h, 4h, 12h
                            'slow_window': [180, 360, 720, 1440, 2880]},  # 3h, 6h, 12h, 1d, 2d},
        'donchian_breakout': { # for 5min bars
                            'lookback':   [240, 720, 1440, 2880, 10080],  # minutes: 4h, 12h, 1d, 2d, 1w
                            'period':     [14*5, 48*5, 288*5],                  # 5m bars: 70 min, 4h, 1d
                            'k':          [2, 3, 5, 8, 12],               # 5m ATR, so larger k for longer lookbacks
                            'chandelier': [False, True],
                            'intrabar': [False, True]
                            }
    },
    '30m': {
        'accurate': {'change_period': [120, 240, 720, 1440],          # 2h, 4h, 12h, 1d
                     'in_cond':       [0.005, 0.01, 0.02],
                     'take_profit':   [1.01, 1.02, 1.03],
                     'stop_loss':     [0.97, 0.98, 0.99]},
        'vol_adjusted': {'change_period': [240, 720, 1440, 4320],     # 4h, 12h, 1d, 3d
                         'vol_window':    [2880, 10080],              # 2d, 1w
                         'z_in':          [1.5, 1.8, 2.1],
                         'tp_sigma':      [1.2, 1.6, 2.0],
                         'sl_sigma':      [1.2, 1.6, 2.0]},
        'trend_reversal': {'fast_window': [240, 720, 1440, 4320],     # 4h, 12h, 1d, 3d
                           'slow_window': [720, 2880, 4320, 10080, 20160],  # 12h .. 2w
                           'in_cond':     [0.5, 1.0, 1.5, 2.0],
                           'out_cond':    [-0.5, 0.0, 0.5]},
        'donchian_breakout': {'lookback':   [720, 1440, 4320, 10080, 20160, 40320],  # 12h .. 4w
                              'period':     [720, 1440, 4320],                       # 12h, 1d, 3d
                              'k':          [4, 7, 11, 17, 25],                      # 30m-ATR units
                              'chandelier': [False, True],
                              'intrabar':   [True]},
    },
    '60m': {
        'accurate': {'change_period': [240, 720, 1440, 4320],         # 4h, 12h, 1d, 3d
                     'in_cond':       [0.01, 0.02, 0.04],
                     'take_profit':   [1.02, 1.03, 1.05],
                     'stop_loss':     [0.95, 0.97, 0.98]},
        'vol_adjusted': {'change_period': [720, 1440, 4320, 10080],   # 12h, 1d, 3d, 1w
                         'vol_window':    [10080, 20160],             # 1w, 2w
                         'z_in':          [1.5, 1.8, 2.1],
                         'tp_sigma':      [1.2, 1.6, 2.0],
                         'sl_sigma':      [1.2, 1.6, 2.0]},
        'trend_reversal': {'fast_window': [720, 1440, 4320, 10080],   # 12h, 1d, 3d, 1w
                           'slow_window': [2880, 4320, 10080, 20160, 40320],  # 2d .. 4w
                           'in_cond':     [0.5, 1.0, 1.5, 2.0],
                           'out_cond':    [-0.5, 0.0, 0.5]},
        'donchian_breakout': {'lookback':   [1440, 4320, 10080, 20160, 40320, 80640],  # 1d .. 8w
                              'period':     [1440, 4320, 10080],                       # 1d, 3d, 1w
                              'k':          [3, 5, 8, 12, 18],                         # 1h-ATR units
                              'chandelier': [False, True],
                              'intrabar':   [True]},
    },

}
# Validity rules per strategy: return False to skip a combination
constraints = {
    'vol_adjusted':      lambda p: p['vol_window'] >= 2 * p['change_period'],
    'trend_reversal':    lambda p: p['slow_window'] >= 3 * p['fast_window']
                                   and p['out_cond'] < p['in_cond'],
    'donchian_breakout': lambda p: p['period'] <= p['lookback'],
}

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
base_eval = {
    'amount': 10000,
    'accumulates': True,
    'cost_bps': 10.0,
    'verbose': False,
    'plot': False,
}
eval_params = {name: dict(base_eval) for name in strat_params}
# %%

if __name__ == '__main__':


    # ---------------------------------------------------------------------------
    # Evaluation settings 5min
    # ---------------------------------------------------------------------------

    from src.data_binance import get_data_binance
    ticker = 'BTC-USD'
    interval = '5m'
    raw_data = get_data_binance(interval=interval)
    lookback = 90 * 24 * 60 // 5 # 60 days in bars
    rebalance_freq = 30 * 24 * 60 // 5 # weekly

    strategy = 'accurate'
    accurate_pnl_5m, accurate_profit_5m, accurate_returns_5m, accurate_cum_returns_5m, _, accurate_logs_5m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'vol_adjusted'
    vol_pnl_5m, vol_profit_5m, vol_returns_5m, vol_cum_returns_5m, _, vol_logs_5m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'trend_reversal'
    trend_pnl_5m, trend_profit_5m, trend_returns_5m, trend_cum_returns_5m, _, trend_logs_5m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'donchian_breakout'
    donchian_pnl_5m, donchian_profit_5m, donchian_returns_5m, donchian_cum_returns_5m, _, donchian_logs_5m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    bench_idx = pd.date_range('2018-01-01 00:00:00+00:00', '2026-09-30 23:55:00+00:00', freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)


    plt.figure(figsize=(12, 6))
    accurate_cum_returns_5m.plot(label='Momentum')
    vol_cum_returns_5m.plot(label='Vol_adjusted Momentum')
    trend_cum_returns_5m.plot(label='EWMA Trend Reversal')
    donchian_cum_returns_5m.plot(label='Donchian Breakout')
    pd.Series(1.10 ** years, index=bench_idx).plot(label='8%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest {interval}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.tight_layout()
    plt.show()

# %%

if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Evaluation settings 5min
    # ---------------------------------------------------------------------------
    interval = '30m'
    raw_data = get_data_binance(interval=interval)
    lookback = 180 * 24 * 60 // 30 # 180 days in bars
    rebalance_freq = 30 * 24 * 60 // 30 # every 30 days

    strategy = 'accurate'
    accurate_pnl_30m, accurate_profit_30m, accurate_returns_30m, accurate_cum_returns_30m, _, accurate_logs_30m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'vol_adjusted'
    vol_pnl_30m, vol_profit_30m, vol_returns_30m, vol_cum_returns_30m, _, vol_logs_30m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'trend_reversal'
    trend_pnl_30m, trend_profit_30m, trend_returns_30m, trend_cum_returns_30m, _, trend_logs_30m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'donchian_breakout'
    donchian_pnl_30m, donchian_profit_30m, donchian_returns_30m, donchian_cum_returns_30m, _, donchian_logs_30m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    bench_idx = pd.date_range('2018-01-01 00:00:00+00:00', '2026-09-30 23:55:00+00:00', freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)


    plt.figure(figsize=(12, 6))
    accurate_cum_returns_30m.plot(label='Momentum')
    vol_cum_returns_30m.plot(label='Vol_adjusted Momentum')
    trend_cum_returns_30m.plot(label='EWMA Trend Reversal')
    donchian_cum_returns_30m.plot(label='Donchian Breakout')
    pd.Series(1.10 ** years, index=bench_idx).plot(label='8%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest {interval}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.tight_layout()
    plt.show()

# %%

if __name__ == '__main__':


    # ---------------------------------------------------------------------------
    # Evaluation settings 360min
    # ---------------------------------------------------------------------------

    from src.data_binance import get_data_binance
    ticker = 'BTC-USD'
    interval = '60m'
    raw_data = get_data_binance(interval=interval)
    lookback = 365 * 24 * 60 // 60 # every 365 days
    rebalance_freq = 60 * 24 * 60 // 60 # every 60 days

    strategy = 'accurate'
    accurate_pnl_60m, accurate_profit_60m, accurate_returns_60m, accurate_cum_returns_60m, _, accurate_logs_60m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'vol_adjusted'
    vol_pnl_60m, vol_profit_60m, vol_returns_60m, vol_cum_returns_60m, _, vol_logs_60m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'trend_reversal'
    trend_pnl_60m, trend_profit_60m, trend_returns_60m, trend_cum_returns_60m, _, trend_logs_60m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    strategy = 'donchian_breakout'
    donchian_pnl_60m, donchian_profit_60m, donchian_returns_60m, donchian_cum_returns_60m, _, donchian_logs_60m = backtest_returns(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategy=strategy,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params[strategy],
        rank_by='sharpe',
        min_trades=20
    )

    bench_idx = pd.date_range('2018-01-01 00:00:00+00:00', '2026-09-30 23:55:00+00:00', freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)


    plt.figure(figsize=(12, 6))
    accurate_cum_returns_60m.plot(label='Momentum')
    vol_cum_returns_60m.plot(label='Vol_adjusted Momentum')
    trend_cum_returns_60m.plot(label='EWMA Trend Reversal')
    donchian_cum_returns_60m.plot(label='Donchian Breakout')
    pd.Series(1.10 ** years, index=bench_idx).plot(label='8%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest {interval}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.tight_layout()
    plt.show()