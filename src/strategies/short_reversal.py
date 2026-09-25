# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.data import raw_data

# %%
def master_short_reversal(
        ticker: str,
        interval: str, 
        lookback: int,
        fit_window: int,
        refit_window: int,
        z_in: float,
        k_halflife: float,
        max_hold: int
    ):
    '''
    Input: OHLCV dataframe for specific ticker, hyperparameters for the basic momentum strategy. Interval is string ('5m') 
           Windows are in minutes.
           In and out_cond are thresholds for difference in ewma averages (in at >0, out at <0 eg)
    Output: Tuple of dataframe, trade tuples (price_in, price_out) and respective dates
    '''

    df = raw_data[ticker]
    df = data_prep_short_reversal(df=df, interval=interval, lookback=lookback, fit_window=fit_window, refit_window=refit_window)
    trades, dates = short_reversal(df=df, z_in=z_in, k_halflife=k_halflife, max_hold=max_hold)

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

def get_std(df: pd.Series) -> tuple:
    '''
    Fit p_t = m_t + u_t with
        m_t : random walk, increment std sigma_m
        u_t : AR(1), u_t = phi * u_{t-1} + eps_t
    by Kalman-filter MLE.

    Input : log price window (pd.Series)
    Output: (sigma_m, sigma_u, phi), in log-return units per bar.
            sigma_u is the STATIONARY std of u_t, not the innovation std.
            Returns NaNs if the fit fails or is non-stationary.
    '''

    y = ((df - df.iloc[0]) * 100).to_numpy() #fit percentage points

    model = sm.tsa.UnobservedComponents(y, level='random walk', autoregressive=1)
    results = model.fit(method='lbfgs', maxiter=4000, disp=False)
    # print(results.mle_retvals['warnflag'])
    p = dict(zip(model.param_names, results.params))

    phi = p['ar.L1']
    # if not results.mle_retvals.get('converged', False) or abs(phi) >= 1:
    #     return np.nan, np.nan, np.nan
    if abs(phi) >= 1:
        return np.nan, np.nan, np.nan

    sigma_m = np.sqrt(p['sigma2.level']) / 100
    sigma_eps = np.sqrt(p['sigma2.ar']) / 100
    sigma_u = sigma_eps / np.sqrt(1 - phi**2)

    return sigma_m, sigma_u, phi

def rolling_params(df: pd.Series, fit_bars: int, refit_bars:int) -> pd.DataFrame:
    '''
    Walk-forward fit: every refit_bars, fit on the trailing fit_bars bars.
    Parameters fitted on bars [end - fit_bars, end - 1] are stamped at bar `end`
    and forward-filled, so bar t only ever uses params from data before t.
    '''   

    params = pd.DataFrame(np.nan, index=df.index, columns=['sigma_m', 'sigma_u', 'phi'])

    for end in range(fit_bars, len(df), refit_bars):
        window = df.iloc[end - fit_bars : end]
        params.iloc[end] = get_std(window)

    return params.ffill()


def data_prep_short_reversal(
        df: pd.DataFrame,  
        interval: str, 
        lookback: int,
        fit_window: int = 7 * 1440,
        refit_window: int = 1440,
    ) -> pd.DataFrame:
    '''
    Short-horizon reversal signal on log prices for a single ticker.

    Model: p_t = m_t + u_t, m_t random walk (sigma_m), u_t AR(1) (phi, sigma_u).
    lookback, fit_window, refit_window are in MINUTES.

    Adds columns:
        R_past    : L-bar log return, L = lookback in bars
        sigma_m   : random-walk increment std per bar (walk-forward fit)
        sigma_u   : stationary std of the transient component
        phi       : AR(1) coefficient of the transient component
        half_life : ln 2 / -ln phi, in bars (NaN if phi <= 0)
        Var_L     : L * sigma_m^2 + 2 * sigma_u^2 * (1 - phi^L)
        z         : R_past / sqrt(Var_L), ~N(0,1) under the model

    Signal at bar t uses data up to and including Close_t;
    trade on bar t+1 in the backtest.
    Bar-count based, so best for tickers without session breaks.
    '''

    new_df = df.copy()
    n_bars = bars_in_period(interval=interval, window=lookback)
    fit_bars = bars_in_period(interval=interval, window=fit_window)
    refit_bars = bars_in_period(interval=interval, window=refit_window)

    log_price = np.log(new_df['Close']) 
    new_df['R_past'] = log_price.diff(n_bars)

    params = rolling_params(log_price, fit_bars=fit_bars, refit_bars=refit_bars)
    new_df[['sigma_m', 'sigma_u', 'phi']] = params

    new_df['halflife'] = np.log(2) / -np.log(new_df['phi'])
    new_df['Var_L'] = n_bars * new_df['sigma_m']**2 + 2 * new_df['sigma_u']**2 * (1 - new_df['phi']**n_bars)
    new_df['z'] = new_df['R_past'] / np.sqrt(new_df['Var_L'])

    return new_df
    

def short_reversal(
        df: pd.DataFrame,
        z_in: float,
        k_halflife: float,
        max_hold: int = 10
    ) -> tuple:
    '''
    Long short-horizon mean reversal strategy with scaled halflife exit.
    Enter when z[i] <= -z_in, exit when after halflife[i] * k
    Decisions use z at the close of bar i and execute at the open of bar i+1.

    Returns:
        trades     : list of (price_in, price_out)
        dates      : list of (date_in, date_out)
    '''

    in_trade = False
    trades = []
    dates = []
    n = len(df)
    o = df['Open'].to_numpy()
    z = df['z'].to_numpy()
    halflife = df['halflife'].to_numpy()

    for i in range(n - 1):
        if in_trade:
            if i < entry_i + hold:
                continue

            in_trade = False
            out_price = o[i]
            trades.append((in_price, out_price))
            dates.append((df.index[entry_i], df.index[i]))

        if np.isnan(z[i]) or np.isnan(halflife[i]) or z[i] > -z_in:
            continue
        
        in_trade = True
        entry_i = i + 1
        in_price = o[entry_i]
        hold = max(1, int(np.ceil(halflife[i] * k_halflife)))
        hold = min(hold, max_hold)
            
    
    return trades, dates

       
# %%

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']  

    hyperparams = {
        'interval': '5m', 
        'lookback': 15,
        'fit_window': 7 * 1440,
        'refit_window': 1440,
        'z_in': 0.42,
        'k_halflife': 1,
        'max_hold': 12
            }
    for ticker in tickers:
        _, _, dates = master_short_reversal(ticker=ticker, **hyperparams)
    diffs = np.array([(t2 - t1).total_seconds() / 60 // 5 for t1, t2 in dates])
    median = np.median(diffs) # 51