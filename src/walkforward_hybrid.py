# %%
import os 
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.strategies.accurate_momentum import master_accurate_momentum
from src.strategies.vol_adjusted_momentum import master_vol_adjusted_momentum
from src.strategies.trend_reversal import master_trend_reversal
from src.strategies.donchian_breakout import master_donchian_breakout

from src.grid_search import get_optimal_params
from src.strat_eval import eval_strat, restrict_to_window
# %%


def backtest_returns_hybrid(
        ticker: str,
        data: dict,
        interval: str,
        strategies: list,
        lookback: int,
        rebalance_freq: int,
        eval_kwargs: dict,
        rank_by: str,
        min_trades: int,
        verbose: bool = True,
    ):
    
    amount = eval_kwargs[strategies[0]]['amount']
    accumulates = eval_kwargs[strategies[0]].get('accumulates', True)
    cost_bps = eval_kwargs[strategies[0]].get('cost_bps', 10.0)

    rolling_trades, rolling_dates, params_log = compute_rolling_params_hybrid(
        ticker=ticker,
        data=data,
        interval=interval,
        strategies=strategies,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_kwargs,
        rank_by=rank_by,
        min_trades=min_trades,
        verbose=verbose,
    )

    rolling_pnl, rolling_returns, _ = rolling_eval_strat_hybrid(
        rolling_trades=rolling_trades,
        amount=amount,
        accumulates=accumulates,
        cost_bps=cost_bps
    )
    
    rolling_returns = np.asarray(rolling_returns, dtype=float)
    if accumulates:
        cum_rolling_returns = np.cumprod(1 + rolling_returns)
    else:
        cum_rolling_returns = 1 + np.cumsum(rolling_returns)
    equity = amount * cum_rolling_returns
    rolling_pnl = np.diff(equity, prepend=amount)
    rolling_profit = equity[-1] - amount if len(equity) else 0.0

    exit_dates = pd.DatetimeIndex([d_out for _, d_out in rolling_dates])
    rolling_returns = pd.Series(rolling_returns, index=exit_dates)
    cum_rolling_returns = pd.Series(cum_rolling_returns, index=exit_dates)

    return rolling_pnl, rolling_profit, rolling_returns, cum_rolling_returns, rolling_dates, params_log
# %%

# # # # # # # # # # # # #
# Rolling portfolio optimizer
# # # # # # # # # # # # #

def compute_rolling_params_hybrid(
        ticker: str,
        data: dict,
        interval: str,
        strategies: list,
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
    
    df = data[ticker]

    rolling_trades, rolling_dates, params_log = [], [], []

    starts = list(range(lookback, len(df), rebalance_freq))
    for i, t in enumerate(starts, 1):
        t_start = df.index[t]
        t_stop = min(t + rebalance_freq, len(df))
        t_end = df.index[t_stop] if t_stop < len(df) else None    # exclusive bound

        train = df.iloc[t - lookback : t].dropna()
        if len(train) < lookback:
            continue

        best_score, best_strategy, best_params = 0, None, None

        for strategy in strategies:
            optimal_params, score = get_optimal_params(
                strategy=strategy,
                ticker=ticker,
                interval=interval,
                data={ticker: train},
                grid=param_grids[interval][strategy],
                eval_kwargs=eval_kwargs[strategy],
                rank_by=rank_by,
                min_trades=min_trades,
                constraint=constraints.get(strategy)
            )

            if optimal_params is not None and score > best_score:
                best_score, best_strategy, best_params = score, strategy, optimal_params

        if best_params is None:
            params_log.append({'start': t_start, 'strategy': None})
            continue
        params_log.append({'start': t_start, 'strategy': best_strategy,
                   'is_score': best_score, 'params': best_params})

        if rank_by == 'sharpe' and best_score == 0: 
            continue
        
        if verbose:
            print(f'[{i}/{len(starts)}] {t_start:%Y-%m-%d}, best strategy: {best_strategy}', flush=True)

        spec = strat_params[best_strategy]
        fn, base_params = spec['fn'], spec['params']
        full_best_params = base_params | best_params

        test = df.iloc[t - lookback : t_stop]
        out = fn(ticker=ticker, interval=interval, data={ticker: test}, **full_best_params)
        test_df, trades, dates = out[:3]

        _, trades, dates = restrict_to_window(test_df, trades, dates, start=t_start, end=t_end)
        if not trades:
            continue

        rolling_trades.append(trades)
        rolling_dates += dates

    params_log = pd.DataFrame(params_log).set_index('start') if params_log else pd.DataFrame()
    return rolling_trades, rolling_dates, params_log


def rolling_eval_strat_hybrid(
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
    'change_period': 240,
    'in_cond': 0.01,
    'take_profit': 1.015,
    'stop_loss': 0.985,
}

strat_params = {
    'accurate':     {'fn': master_accurate_momentum, 'params': barrier_params},
    'vol_adjusted': {'fn': master_vol_adjusted_momentum, 'params': {
        'change_period': 240,
        'z_in': 1.8,
        'tp_sigma': 1.6,
        'sl_sigma': 1.6,
        'vol_window': 576*5,
        'tau_mult': 3.0,
        'min_sigma': 0.0,
    }},
    'trend_reversal': {'fn': master_trend_reversal, 'params': {
        'fast_window': 60,
        'slow_window': 180,
        'in_cond': 1.0,
        'out_cond': -0.5,
        'burn_spans': 3,
    }},
    'donchian_breakout': {'fn': master_donchian_breakout, 'params':{ 
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
    '1d': {
        'accurate': {'change_period': [4320, 7200, 14400, 28800],    # 3d, 5d, 10d, 20d
                     'in_cond':       [0.03, 0.05, 0.10],
                     'take_profit':   [1.05, 1.10, 1.20],
                     'stop_loss':     [0.90, 0.93, 0.95]},
        'vol_adjusted': {'change_period': [4320, 7200, 14400, 28800],  # 3d, 5d, 10d, 20d
                         'vol_window':    [43200, 86400],              # 30d, 60d
                         'z_in':          [1.5, 1.8, 2.1],
                         'tp_sigma':      [1.2, 1.6, 2.0],
                         'sl_sigma':      [1.2, 1.6, 2.0]},
        'trend_reversal': {'fast_window': [4320, 7200, 14400, 28800],      # 3d, 5d, 10d, 20d
                           'slow_window': [28800, 57600, 86400, 144000],   # 20d, 40d, 60d, 100d
                           'in_cond':     [0.5, 1.0, 1.5, 2.0],
                           'out_cond':    [-0.5, 0.0, 0.5]},
        'donchian_breakout': {'lookback':   [14400, 28800, 79200, 144000],  # 10d, 20d, 55d, 100d
                              'period':     [14400, 20160, 28800],          # ATR: 10d, 14d, 20d
                              'k':          [2, 3, 4, 6],                   # daily-ATR units
                              'chandelier': [False, True],
                              'intrabar':   [True]},
    }

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
cost_bps = 10.0
base_eval = {
    'amount': 10000,
    'accumulates': True,
    'cost_bps': cost_bps,
    'verbose': False,
    'plot': False,
}
eval_params = {name: dict(base_eval) for name in strat_params}
strategies = ['accurate', 'vol_adjusted', 'trend_reversal', 'donchian_breakout']
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

    hybrid_pnl_5m, hybrid_profit_5m, hybrid_returns_5m, hybrid_cum_returns_5m, _, hybrid_logs_5m = backtest_returns_hybrid(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategies=strategies,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params,
        rank_by='sharpe',
        min_trades=20
    )

    
# %%
    # load the per-strategy cum returns saved earlier
    cum_df = pd.read_csv(f'results/walkforward_cum_returns_{interval}.csv',
                         index_col='exit_time', parse_dates=True)
    cum_df.index = pd.to_datetime(cum_df.index, utc=True)      # ensure tz-aware UTC, to match the hybrid

    # add the hybrid as a new column (outer join on exit time)
    hybrid = hybrid_cum_returns_5m.groupby(level=0).last().rename('hybrid')
    combined_df = cum_df.join(hybrid, how='outer').sort_index()
    combined_df.index.name = 'exit_time'

    combined_df.to_csv(f'results/hybrid_walkforward_cum_returns_{interval}.csv')

    # plot everything together
    labels = {
        'accurate':          'Momentum',
        'vol_adjusted':      'Vol_adjusted Momentum',
        'trend_reversal':    'EWMA Trend Reversal',
        'donchian_breakout': 'Donchian Breakout',
        'hybrid':            'Hybrid',
    }

    bench_idx = pd.date_range(raw_data[ticker].index[lookback], raw_data[ticker].index[-1], freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)
    rfr = 1.10

    plt.figure(figsize=(12, 6))
    for col in combined_df.columns:
        style = {'lw': 2.2, 'color': 'black'} if col == 'hybrid' else {'lw': 1.2}
        combined_df[col].dropna().plot(label=labels.get(col, col), **style)
    pd.Series(rfr ** years, index=bench_idx).plot(
        label=f'{int((rfr - 1) * 100)}%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest for {interval} bars and cost_bps={cost_bps}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_all_walkforward_{interval}_{cost_bps}bps.png',
                dpi=150, bbox_inches='tight')
    plt.show()

# %%

if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Evaluation settings 30min
    # ---------------------------------------------------------------------------
    interval = '30m'
    raw_data = get_data_binance(interval=interval)
    lookback = 180 * 24 * 60 // 30 # 180 days in bars
    rebalance_freq = 30 * 24 * 60 // 30 # every 30 days

    hybrid_pnl_30m, hybrid_profit_30m, hybrid_returns_30m, hybrid_cum_returns_30m, _, hybrid_logs_30m = backtest_returns_hybrid(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategies=strategies,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params,
        rank_by='sharpe',
        min_trades=20
    )


# %%
    # load the per-strategy cum returns saved earlier
    cum_df = pd.read_csv(f'results/walkforward_cum_returns_{interval}.csv',
                         index_col='exit_time', parse_dates=True)
    cum_df.index = pd.to_datetime(cum_df.index, utc=True)      # ensure tz-aware UTC, to match the hybrid

    # add the hybrid as a new column (outer join on exit time)
    hybrid = hybrid_cum_returns_30m.groupby(level=0).last().rename('hybrid')
    combined_df = cum_df.join(hybrid, how='outer').sort_index()
    combined_df.index.name = 'exit_time'

    combined_df.to_csv(f'results/hybrid_walkforward_cum_returns_{interval}.csv')

    # plot everything together
    labels = {
        'accurate':          'Momentum',
        'vol_adjusted':      'Vol_adjusted Momentum',
        'trend_reversal':    'EWMA Trend Reversal',
        'donchian_breakout': 'Donchian Breakout',
        'hybrid':            'Hybrid',
    }

    bench_idx = pd.date_range(raw_data[ticker].index[lookback], raw_data[ticker].index[-1], freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)
    rfr = 1.10

    plt.figure(figsize=(12, 6))
    for col in combined_df.columns:
        style = {'lw': 2.2, 'color': 'black'} if col == 'hybrid' else {'lw': 1.2}
        combined_df[col].dropna().plot(label=labels.get(col, col), **style)
    pd.Series(rfr ** years, index=bench_idx).plot(
        label=f'{int((rfr - 1) * 100)}%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest for {interval} bars and cost_bps={cost_bps}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_all_walkforward_{interval}_{cost_bps}bps.png',
                dpi=150, bbox_inches='tight')
    plt.show()

# %%

if __name__ == '__main__':


    # ---------------------------------------------------------------------------
    # Evaluation settings 60min
    # ---------------------------------------------------------------------------

    from src.data_binance import get_data_binance
    ticker = 'BTC-USD'
    interval = '60m'
    raw_data = get_data_binance(interval=interval)
    lookback = 365 * 24 * 60 // 60 # every 365 days
    rebalance_freq = 60 * 24 * 60 // 60 # every 60 days

    hybrid_pnl_60m, hybrid_profit_60m, hybrid_returns_60m, hybrid_cum_returns_60m, _, hybrid_logs_60m = backtest_returns_hybrid(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategies=strategies,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params,
        rank_by='sharpe',
        min_trades=20
    )

# %%
    # load the per-strategy cum returns saved earlier
    cum_df = pd.read_csv(f'results/walkforward_cum_returns_{interval}.csv',
                         index_col='exit_time', parse_dates=True)
    cum_df.index = pd.to_datetime(cum_df.index, utc=True)      # ensure tz-aware UTC, to match the hybrid

    # add the hybrid as a new column (outer join on exit time)
    hybrid = hybrid_cum_returns_60m.groupby(level=0).last().rename('hybrid')
    combined_df = cum_df.join(hybrid, how='outer').sort_index()
    combined_df.index.name = 'exit_time'

    combined_df.to_csv(f'results/hybrid_walkforward_cum_returns_{interval}.csv')

    # plot everything together
    labels = {
        'accurate':          'Momentum',
        'vol_adjusted':      'Vol_adjusted Momentum',
        'trend_reversal':    'EWMA Trend Reversal',
        'donchian_breakout': 'Donchian Breakout',
        'hybrid':            'Hybrid',
    }

    bench_idx = pd.date_range(raw_data[ticker].index[lookback], raw_data[ticker].index[-1], freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)
    rfr = 1.10

    plt.figure(figsize=(12, 6))
    for col in combined_df.columns:
        style = {'lw': 2.2, 'color': 'black'} if col == 'hybrid' else {'lw': 1.2}
        combined_df[col].dropna().plot(label=labels.get(col, col), **style)
    pd.Series(rfr ** years, index=bench_idx).plot(
        label=f'{int((rfr - 1) * 100)}%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest for {interval} bars and cost_bps={cost_bps}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_all_walkforward_{interval}_{cost_bps}bps.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# %%

if __name__ == '__main__':


    # ---------------------------------------------------------------------------
    # Evaluation settings 1d
    # ---------------------------------------------------------------------------

    from src.data import get_data_yf
    ticker = 'BTC-USD'
    interval = '1d'
    raw_data = get_data_yf(interval=interval)
    lookback = 730 # every 2years
    rebalance_freq = 60 # every 90 days

    strategy = 'accurate'
    hybrid_pnl_1d, hybrid_profit_1d, hybrid_returns_1d, hybrid_cum_returns_1d, _, hybrid_logs_1d = backtest_returns_hybrid(
        ticker=ticker,
        data=raw_data,
        interval=interval,
        strategies=strategies,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        eval_kwargs=eval_params,
        rank_by='sharpe',
        min_trades=10
    )

# %%
    # load the per-strategy cum returns saved earlier
    cum_df = pd.read_csv(f'results/walkforward_cum_returns_{interval}.csv',
                         index_col='exit_time', parse_dates=True)
    cum_df.index = pd.to_datetime(cum_df.index, utc=True)      # ensure tz-aware UTC, to match the hybrid

    # add the hybrid as a new column (outer join on exit time)
    hybrid = hybrid_cum_returns_1d.groupby(level=0).last().rename('hybrid')
    combined_df = cum_df.join(hybrid, how='outer').sort_index()
    combined_df.index.name = 'exit_time'

    combined_df.to_csv(f'results/hybrid_walkforward_cum_returns_{interval}.csv')

    # plot everything together
    labels = {
        'accurate':          'Momentum',
        'vol_adjusted':      'Vol_adjusted Momentum',
        'trend_reversal':    'EWMA Trend Reversal',
        'donchian_breakout': 'Donchian Breakout',
        'hybrid':            'Hybrid',
    }

    bench_idx = pd.date_range(raw_data[ticker].index[lookback], raw_data[ticker].index[-1], freq='1D')
    years = (bench_idx - bench_idx[0]) / pd.Timedelta(days=365.25)
    rfr = 1.10

    plt.figure(figsize=(12, 6))
    for col in combined_df.columns:
        style = {'lw': 2.2, 'color': 'black'} if col == 'hybrid' else {'lw': 1.2}
        combined_df[col].dropna().plot(label=labels.get(col, col), **style)
    pd.Series(rfr ** years, index=bench_idx).plot(
        label=f'{int((rfr - 1) * 100)}%/yr benchmark', ls='--', color='grey')
    plt.legend()
    plt.title(f'Walkforward backtest for {interval} bars and cost_bps={cost_bps}')
    plt.xticks(rotation=5)
    plt.grid()
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_all_walkforward_{interval}_{cost_bps}bps.png',
                dpi=150, bbox_inches='tight')
    plt.show()
