# %%
import itertools
from typing import Callable
import traceback

import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis

from src.strategies.basic_momentum import master_basic_momentum
from src.strategies.refined_momentum import master_refined_momentum
from src.strategies.accurate_momentum import master_accurate_momentum
from src.strategies.vol_adjusted_momentum import master_vol_adjusted_momentum
from src.strategies.momentum_fixed_horizon import master_momentum_fixed_horizon
from src.strategies.trend_reversal import master_trend_reversal
from src.strategies.short_reversal import master_short_reversal
from src.strategies.donchian_breakout import master_donchian_breakout
from src.strat_eval import strat_data

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
    'basic':        {'fn': master_basic_momentum,    'params': barrier_params},
    'refined':      {'fn': master_refined_momentum,  'params': barrier_params},
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
    'fixed_horizon': {'fn': master_momentum_fixed_horizon, 'params': {
        'change_period': 240,
        'z_in': 1.8,
        'horizon': 22*5,
    }},
    'trend_reversal': {'fn': master_trend_reversal, 'params': {
        'fast_window': 60,
        'slow_window': 180,
        'in_cond': 1.0,
        'out_cond': -0.5,
        'burn_spans': 3,
    }},
    'short_reversal': {'fn': master_short_reversal, 'params':{
        'lookback': 15,
        'fit_window': 7 * 1440,
        'refit_window': 1440 // 2,
        'z_in': 3,
        'k_halflife': 2.5,
        'max_hold': float('inf')
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
    # 'donchian_breakout': {'lookback':   [720, 1440, 2880],  # minutes: 4h, 12h, 1d, 2d, 1w
    #                         'period':     [288*5, 576*5, 1152*5],                  # 5m bars: 70 min, 4h, 1d
    #                         'k':          [8, 12, 16, 24, 32],               # 5m ATR, so larger k for longer lookbacks
    #                         'chandelier': [False, True],
    #                         'intrabar': [False, True]
    #                         }
    # 'donchian_breakout': { #for 60min bars
    #                         'lookback': [1440, 4320, 10080, 20160, 40320, 80640],  # 1d, 3d, 1w, 2w, 4w, 8w
    #                         'period':   [1440, 4320, 10080],                       # ATR: 1d, 3d, 1w  (24, 72, 168 bars)
    #                         'k':        [3, 5, 8, 12, 18],                         # in 1h-ATR units
    #                         'chandelier': [False, True],
    #                         'intrabar':   [True],                                  # fix it, don't search it (see below)
    #                     },
}

# Validity rules per strategy: return False to skip a combination
constraints: dict[str, Callable[[dict], bool]] = {
    'trend_reversal': lambda p: p['out_cond'] < p['in_cond']
                                and p['fast_window'] < p['slow_window'],
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

# Silence per-run output during a grid search
GRID_EVAL_OVERRIDES = {'plot': False, 'verbose': False}


TICKER = 'BTC-USD'
RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
MIN_TRADES = 20               # don't rank configs with too few trades to judge
TOP_N = 5


# %%
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unpack(out: tuple) -> tuple:
    '''
    Normalise master_* output to (df, trades, dates, n_ambiguous).
    Accepts (df, trades, dates) or (df, trades, dates, x), where x is used as
    n_ambiguous only if it is an int 
    '''
    df, trades, dates = out[:3]
    extra = out[3] if len(out) > 3 else None
    n_ambiguous = extra if isinstance(extra, int) else 0
    return df, trades, dates, n_ambiguous


def _metrics(strat_out: tuple, df: pd.DataFrame) -> dict:
    '''
    Build ranking metrics from strat_data's
    (trade_pnls, total_pnl, equity_after_each_trade, final_multiple).
    '''
    pnls, total_pnl, equity, final_mult, *_ = strat_out
    pnls = np.asarray(pnls, dtype=float)
    equity = np.asarray(equity, dtype=float)
    n = len(pnls)

    years = (df.index[-1] - df.index[0]) / pd.Timedelta(days=365.25)
    out = {
        'total_pnl': float(total_pnl),
        'total_return': float(final_mult) - 1,
        'cagr': float(final_mult) ** (1 / years) - 1 if years > 0 else np.nan,
    }
    if n == 0:
        return {**out, 'sharpe': np.nan, 'sharpe_per_trade': np.nan, 'win_rate': np.nan,
                'avg_trade_ret': np.nan, 'profit_factor': np.nan, 'max_drawdown': 0.0}

    curve = np.concatenate([[1.0], equity])        # equity before first trade = 1
    rets = curve[1:] / curve[:-1] - 1              # per-trade returns
    sd = rets.std(ddof=1) if n > 1 else np.nan
    sharpe_trade = rets.mean() / sd if sd > 0 else np.nan

    gains, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    return {
        **out,
        'sharpe': sharpe_trade * np.sqrt(n / years) if years > 0 else np.nan,
        'sharpe_per_trade': sharpe_trade,
        'win_rate': (pnls > 0).mean(),
        'avg_trade_ret': rets.mean(),
        'profit_factor': gains / losses if losses > 0 else np.inf,
        'max_drawdown': (curve / np.maximum.accumulate(curve) - 1).min(),
    }


# %%
# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(
        strategy: str,
        ticker: str,
        data: dict,
        interval: str,
        grid: dict[str, list],
        eval_kwargs: dict,
        constraint: Callable[[dict], bool] | None = None,
        verbose: bool = True
    ) -> pd.DataFrame:
    spec = strat_params[strategy]
    fn, base = spec['fn'], spec['params']
    eval_kw = {**eval_kwargs, **GRID_EVAL_OVERRIDES}

    unknown = set(grid) - set(base)
    if unknown:
        raise KeyError(f'Grid keys not in {strategy} params: {sorted(unknown)}')

    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    rows = []

    for k, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values))
        params = {**base, **overrides}
        if constraint is not None and not constraint(params):
            continue

        if verbose:
            print(f'[{k}/{len(combos)}] {overrides}')
        try:
            df, trades, dates, n_ambiguous = _unpack(fn(ticker=ticker, data=data, interval=interval, **params))
            metrics = strat_data(
                ticker, df, trades, dates,
                strategy=strategy,
                n_ambiguous=n_ambiguous,
                **eval_kw,
            )
            rows.append({
                **overrides,
                'n_trades': len(trades),
                'n_ambiguous': n_ambiguous,
                **_metrics(metrics, df),
            })
        except Exception as e:                     # keep the grid running
            print(f'    failed: {type(e).__name__}: {e}')
            if not any('error' in r for r in rows):
                traceback.print_exc()              # full traceback for the first failure only
            rows.append({**overrides, 'error': f'{type(e).__name__}: {e}'})

    return pd.DataFrame(rows)


def summarise(
        results: pd.DataFrame,
        grid_keys: list[str],
        rank_by: str = RANK_BY,
        min_trades: int = MIN_TRADES,
        top: int = TOP_N,
    ) -> pd.DataFrame:
    if 'error' in results.columns:
        failed = results['error'].notna()
        if failed.any():
            print(f'{failed.sum()}/{len(results)} runs failed:')
            print(results.loc[failed, 'error'].value_counts().to_string())
        results = results.loc[~failed].drop(columns='error')
        if results.empty:
            raise RuntimeError('All grid runs failed; see errors above.')
    if rank_by not in results.columns:
        raise KeyError(f"'{rank_by}' not in results. Available: {list(results.columns)}")

    ok = results.dropna(subset=[rank_by])
    eligible = ok[ok['n_trades'] >= min_trades]
    ranked = eligible.sort_values(rank_by, ascending=False)
    fmt = '{:.3f}'.format

    print(f'\nTop {top} by {rank_by} (n_trades >= {min_trades}):')
    print(ranked.head(top).to_string(index=False, float_format=fmt))

    if len(grid_keys) == 2:
        # full surface: look for a plateau, not an isolated peak
        r, c = grid_keys
        print(f'\n{rank_by} grid (rows: {r}, cols: {c}):')
        print(ok.pivot_table(index=r, columns=c, values=rank_by).to_string(float_format=fmt))
    else:
        # marginal means: how sensitive the metric is to each parameter
        for key in grid_keys:
            print(f'\nMean {rank_by} by {key}:')
            print(ok.groupby(key)[rank_by].agg(['mean', 'std', 'count'])
                    .to_string(float_format=fmt))

    return ranked

# %%
# ---------------------------------------------------------------------------
# Get optimal params for walkforward
# ---------------------------------------------------------------------------

def get_optimal_params(
        strategy: str,
        ticker: str,
        data: dict,
        interval: str,
        grid: dict[str, list],
        eval_kwargs: dict,
        rank_by: str,
        min_trades: int,
        rank_by_strat='total_return',
        constraint: Callable[[dict], bool] | None = None
    ) -> dict | None:

    results = grid_search(
            strategy=strategy,
            ticker=ticker,
            interval=interval,
            data=data,
            grid=grid,
            eval_kwargs=eval_kwargs,
            constraint=constraint,
            verbose=False
        )

    if results.empty or rank_by not in results.columns:
        return None, -np.inf, -np.inf
    if 'error' in results.columns:
        results = results[results['error'].isna()]

    ok = results.dropna(subset=[rank_by])
    eligible = ok[ok['n_trades'] >= min_trades]
    if eligible.empty:
        return None, -np.inf, -np.inf  
    
    ranked = eligible.sort_values(rank_by, ascending=False)
    params = {param : ranked[param].iloc[0] for param in list(grid)}
    best = ranked.iloc[0]

    return params, float(best[rank_by]), float(best[rank_by_strat])
# %%
if __name__ == '__main__':    
    from src.data import get_data_yf
    interval = '5m'
    RAW_DATA = get_data_yf(interval=interval)
# %%
if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # Get optimal params example
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'donchian_breakout'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    optimal_params_donchian, metric, ret = get_optimal_params(
        strategy=STRATEGY,
        ticker=TICKER,
        interval=interval,
        data=RAW_DATA,
        grid=param_grids[STRATEGY],
        eval_kwargs=eval_params[STRATEGY],
        rank_by = 'sharpe',
        min_trades=20,
        constraint=constraints.get(STRATEGY)
    )
# %%
if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Donchian breakout Configuration
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'donchian_breakout'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    grid = param_grids[STRATEGY]
    results = grid_search(
        strategy=STRATEGY,
        ticker=TICKER,
        data=RAW_DATA,
        interval=interval,
        grid=grid,
        eval_kwargs=eval_params[STRATEGY],
        constraint=constraints.get(STRATEGY)
    )
    # results.to_csv(f'grid_{STRATEGY}_{TICKER}.csv', index=False)
    ranked = summarise(results, grid_keys=list(grid))

# %%
if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Short Reversal Configuration
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'short_reversal'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    grid = param_grids[STRATEGY]
    results = grid_search(
        strategy=STRATEGY,
        ticker=TICKER,
        data=RAW_DATA,
        interval=interval,
        grid=grid,
        eval_kwargs=eval_params[STRATEGY],
        constraint=constraints.get(STRATEGY),
    )
    # results.to_csv(f'grid_{STRATEGY}_{TICKER}.csv', index=False)
    ranked = summarise(results, grid_keys=list(grid))
# %%
if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Trend Reversal Configuration
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'trend_reversal'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    grid = param_grids[STRATEGY]
    results = grid_search(
        strategy=STRATEGY,
        ticker=TICKER,
        data=RAW_DATA,
        interval=interval,
        grid=grid,
        eval_kwargs=eval_params[STRATEGY],
        constraint=constraints.get(STRATEGY),
    )
    # results.to_csv(f'grid_{STRATEGY}_{TICKER}.csv', index=False)
    ranked = summarise(results, grid_keys=list(grid))

# %%
if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Accurate Momentum Reversal Configuration
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'accurate'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    grid = param_grids[STRATEGY]
    results = grid_search(
        strategy=STRATEGY,
        ticker=TICKER,
        data=RAW_DATA,
        interval=interval,
        grid=grid,
        eval_kwargs=eval_params[STRATEGY],
        constraint=constraints.get(STRATEGY),
    )
    # results.to_csv(f'grid_{STRATEGY}_{TICKER}.csv', index=False)
    ranked = summarise(results, grid_keys=list(grid))

# %%
if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Vol adjusted Momentum Reversal Configuration
    # ---------------------------------------------------------------------------
    TICKER = 'BTC-USD'
    STRATEGY = 'vol_adjusted'   # any key of strat_params
    RANK_BY = 'sharpe'            # must match a metric key returned by strat_data
    MIN_TRADES = 20               # don't rank configs with too few trades to judge
    TOP_N = 5

    grid = param_grids[STRATEGY]
    results = grid_search(
        strategy=STRATEGY,
        ticker=TICKER,
        data=RAW_DATA,
        interval=interval,
        grid=grid,
        eval_kwargs=eval_params[STRATEGY],
        constraint=constraints.get(STRATEGY),
    )
    # results.to_csv(f'grid_{STRATEGY}_{TICKER}.csv', index=False)
    ranked = summarise(results, grid_keys=list(grid))