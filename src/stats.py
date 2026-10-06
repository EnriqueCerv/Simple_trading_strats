# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.walkforward_hybrid import param_grids
from math import prod
from src.data_binance import get_data_binance
from src.data import get_data_yf
# %%
def summary_table(
        strategies: list, 
        interval: str,
        lookback: int,
        benchmark: str = 'BTC-USD', 
        n_boot: int = 10000
) -> pd.DataFrame:

    rows = {}
    bootstrap_sharpes = {}

    all_returns = pd.read_csv(f'results/hybrid_walkforward_returns_{interval}.csv',
                         index_col='exit_time', parse_dates=True)

    total_settings = sum(prod(len(value) for value in param_grids[interval][strategy].values()) for strategy in strategies if strategy != 'BTC-USD')
    all_returns.index = pd.to_datetime(all_returns.index, utc=True)

    if interval == '1d':
        prices = get_data_yf(interval=interval)[benchmark]
        prices.index = pd.to_datetime(prices.index, utc=True)
    else:
        prices = get_data_binance(interval=interval)[benchmark]
    oos_start, oos_end = prices.index[lookback], prices.index[-1]
    benchmark_bar_returns = prices['Close'].pct_change()

    for strategy in strategies:
        if strategy == benchmark:
            returns = benchmark_bar_returns.dropna()
        else:
            returns = all_returns[strategy].dropna()
        daily_returns = to_daily_returns(returns, start=oos_start, end=oos_end)

        stats = performance_stats(returns=daily_returns)
        boot = block_bootstrap_sharpe(daily_returns, n_boot=n_boot)
        stats['Sharpe CI lower'] = boot['ci_lower']
        stats['Sharpe CI upper'] = boot['ci_upper']
        bootstrap_sharpes[strategy] = boot['bootstrap_distribution']

        sharpe = daily_returns.mean() / daily_returns.std()
        stats['DSR'] = deflated_sharpe(
            observed_sharpe=sharpe,
            n_trials=total_settings,
            t_days=len(daily_returns),
            skew=stats['Skew'],
            kurtosis=stats['Kurtosis']
        )

        if strategy == benchmark:
            stats['Mean Sharpe diff'] = 0
            stats['Sharpe diff CI lower'] = np.nan
            stats['Sharpe diff CI upper'] = np.nan
            stats['Pct wins vs Benchmark'] = np.nan

        else:
            benchmark_returns = to_daily_returns(benchmark_bar_returns, start=oos_start, end=oos_end)
            boot_diff = sharpe_difference_test(returns1=returns, returns2=benchmark_returns, n_boot=n_boot)
            stats['Mean Sharpe diff'] = boot_diff['mean_diff']
            stats['Sharpe diff CI lower'] = boot_diff['ci_lower']
            stats['Sharpe diff CI upper'] = boot_diff['ci_upper']
            stats['Pct wins vs Benchmark'] = boot_diff['pct_returns1_wins']

        rows[strategy] = stats

    return pd.DataFrame(rows).T, bootstrap_sharpes
# %%

def to_daily_returns(trade_returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    '''Per-trade returns (indexed by exit time) -> daily returns on [start, end], 0 on days with no exit.'''
    daily = (1 + trade_returns).groupby(trade_returns.index.floor('D')).prod() - 1
    days = pd.date_range(start.floor('D'), end.floor('D'), freq='D')
    return daily.reindex(days, fill_value=0.0)


def performance_stats(returns: pd.Series) -> pd.Series:
    '''
    Outputs: Annual Return, Annual Vol, Sharpe, Max Drawdown, Calmar,
                Total time underwater, Max time underwater, Avg Turnover (annualized), Skew, Kurtosis
    '''
    annual_return = (1 + returns).prod() ** (365 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(365)
    sharpe = returns.mean() / returns.std() * np.sqrt(365)

    cum_returns = (1 + returns).cumprod()

    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()

    is_underwater = cum_returns < cum_returns.cummax()
    time_underwater = is_underwater.astype(int).groupby(
        (is_underwater != is_underwater.shift()).cumsum()
    ).sum().max()

    return pd.Series({
        'Annual Return': annual_return,
        'Annual Vol': annual_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown,
        'Calmar': annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan,
        'Time Underwater': time_underwater,
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })

def block_bootstrap_sharpe(
        returns: pd.Series,
        n_boot: int = 10000,
        block_size: int = 21, 
        ci: float = 95.0,
        seed: int = 42
) -> dict:
    '''
    Outputs: {'sharpe': float, 'ci_lower': float, 'ci_upper': float,
              'bootstrap_distribution': np.array}
    '''

    n = len(returns)
    returns_arr = returns.values
    sharpes = np.zeros(n_boot)
    np.random.seed(seed)

    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    samples = returns_arr[indices]
    sharpes = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(365)

    return {
        'Sharpe': (returns.mean() / returns.std()) * np.sqrt(365),
        'ci_lower': np.percentile(sharpes, (100 - ci) / 2),
        'ci_upper': np.percentile(sharpes, (100 + ci) / 2),
        'bootstrap_distribution': sharpes
    }


def sharpe_difference_test(returns1: pd.Series, returns2: pd.Series,
                           start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                           n_boot: int = 10000, block_size: int = 21,
                           ci: float = 95, periods_per_year: int = 365,
                           seed: int = 42) -> dict:
    '''
    Paired circular block bootstrap of the annualised Sharpe difference SR1 - SR2.
    returns1, returns2: per-trade net returns indexed by exit time (tz-aware).
    start, end: common evaluation window; pass the walk-forward oos_start and data end
                so both strategies are measured over the same days (cash before first trade).
    '''
    start = start if start is not None else min(returns1.index.min(), returns2.index.min())
    end = end if end is not None else max(returns1.index.max(), returns2.index.max())

    x = np.column_stack([to_daily_returns(returns1, start, end).values,
                         to_daily_returns(returns2, start, end).values])   # (n, 2)
    n = len(x)

    def sharpe(a, axis):
        sd = a.std(axis=axis, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(sd > 0, a.mean(axis=axis) / sd, np.nan) * np.sqrt(periods_per_year)

    observed = sharpe(x, axis=0)                                        # (2,)
    observed_diff = observed[0] - observed[1]

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))                # (n_boot, n_blocks)
    offsets = np.arange(block_size)                                     # (block_size,)
    indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(n_boot, -1)[:, :n]

    samples = x[indices]                                                # (n_boot, n, 2)
    boot = sharpe(samples, axis=1)                                      # (n_boot, 2)
    diff = boot[:, 0] - boot[:, 1]
    diff = diff[~np.isnan(diff)]                                        # drop resamples with zero variance

    return {
        'sharpe1': observed[0],
        'sharpe2': observed[1],
        'observed_diff': observed_diff,
        'mean_diff': diff.mean(),
        'ci_lower': np.percentile(diff, (100 - ci) / 2),
        'ci_upper': np.percentile(diff, (100 + ci) / 2),
        'pct_returns1_wins': (diff > 0).mean(),
        'n_days': n,
        'n_boot_valid': len(diff),
    }


def deflated_sharpe(
        observed_sharpe: float, 
        n_trials: int, 
        t_days: int, 
        skew: float, 
        kurtosis: float
) -> float:
    '''
    Input: daily (not annualized) sharpe, number of allocators tried (5), t_days = len(returns)
    Returns: probability that the observed Sharpe is significant after
             multiple-testing correction (Bailey & López de Prado, 2014).
    '''

    em = 0.5772
    var_sharpe = (1 / t_days) * (1 - skew * observed_sharpe + (kurtosis + 2) / 4 * observed_sharpe ** 2)
    sharpe_null = np.sqrt(var_sharpe) * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * np.exp(1))))
    dsr = (observed_sharpe - sharpe_null) * np.sqrt(t_days)
    dsr /= np.sqrt(1 - skew * observed_sharpe + (kurtosis + 2) / 4 * observed_sharpe ** 2)

    return norm.cdf(dsr)
# %%
if __name__ == '__main__':
    interval = '1d'
    strategies = ['BTC-USD', 'accurate', 'vol_adjusted', 'trend_reversal', 'donchian_breakout']
    lookbacks = {
        '5m': 90 * 24 * 60 // 5,
        '30m': 180 * 24 * 60 // 30,
        '60m': 365 * 24 * 60 // 60,
        '1d': 730
    }

    rows, sharpes = summary_table(strategies=strategies, interval=interval, lookback=lookbacks[interval])
    rows