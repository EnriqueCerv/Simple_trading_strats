# %%
"""
Simple momentum strategy, intraday version with OHLC-aware exits.

Rule (from the iftrades screenshot):
    IF price moves up  `in_cond`  over `change_period` minutes  -> BUY `amount` USD
    IF price above purchase * out_up_cond  OR  below purchase * out_down_cond -> SELL

Execution model
---------------
Entry : signal is read on the CLOSE of bar i, filled at the OPEN of bar i+1.
Exit  : take-profit / stop-loss are treated as RESTING ORDERS, so they fill
        intrabar at the barrier level, not at the bar close.
Forced: any position still open on the last bar of a session is flattened at
        that bar's OPEN (see `basic_strat` for why the open and not the close).

Sessions
--------
`continuous=False` (equities): momentum is computed within the calendar day and
positions are never carried overnight.
`continuous=True` (e.g. BTC-USD): no session boundaries at all.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf


# %%
def parse_interval(interval):
    """'5m' -> 5, '1h' -> 60. Minutes, as an int."""
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == 'm':
        return value
    if unit == 'h':
        return value * 60
    if unit == 'd':
        return value * 60 * 24
    raise ValueError(f'unsupported interval: {interval!r}')


def load(ticker, interval, period, start=None, end=None, plot=True):
    df = yf.Ticker(ticker=ticker)
    if start is not None:
        df = df.history(start=start, end=end, interval=interval)
    else:
        df = df.history(period=period, interval=interval)

    if plot:
        mpf.plot(df, type='candle', style='yahoo', title=f'{ticker}')

    return df


# %%
def data_prep(df, interval, change_period, continuous=False):
    """Add `pct_change` (the momentum signal) and `is_last` (session boundary).

    This is the ONLY place the instrument's session structure enters.
    `basic_strat` stays agnostic.
    """
    df = df.copy()
    n_bars = change_period // parse_interval(interval)
    if n_bars < 1:
        raise ValueError('change_period is shorter than one bar')

    if continuous:
        # 24/7 instrument: the bar index is a uniform lattice over time, so a
        # plain n-bar lookback really is `change_period` minutes.
        df['pct_change'] = df['Close'].pct_change(n_bars)
        df['is_last'] = False
    else:
        # Session-bound instrument: an n-bar lookback that straddles the close
        # would silently fold the overnight gap into "intraday momentum".
        # Grouping by date makes the first n_bars of each day NaN, which is the
        # correct answer -- at 10:00 you genuinely do not have an hour of
        # intraday history yet.
        by_day = df.groupby(df.index.date)
        df['pct_change'] = by_day['Close'].pct_change(n_bars)
        df['is_last'] = df.index.isin(by_day.tail(1).index)

    # Always flag the final bar of the dataset so that (a) any open position is
    # closed out rather than silently dropped, and (b) the loop never reaches
    # past the end of the arrays. Needed explicitly in the continuous case.
    df.iloc[-1, df.columns.get_loc('is_last')] = True

    # NOTE: `is_last` is derived from the data, not from an exchange calendar.
    # On a half-day or a session with missing bars, "last row present" is only
    # knowable after the fact. Fine for a backtest, not for a live run.
    return df


# %%
def basic_strat(df, in_cond, out_up_cond, out_down_cond, amount, cost_bps=0.0):
    """Return (trades, gain, n_trades, forced, n_ambiguous, n_gap).

    trades       : list, realised P/L per trade in USD (net of cost_bps)
    gain         : float, sum of `trades`
    n_trades     : int, number of closed trades
    forced       : int, exits caused by the session close rather than a barrier
    n_ambiguous  : int, bars where BOTH barriers were touched (see below)
    n_gap        : int, exits where the bar OPENED past a barrier

    cost_bps is charged per leg on notional, so a 5bp broker fee each way is
    cost_bps=5 (charged twice, once on entry notional and once on exit).
    """
    op = df['Open'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    close = df['Close'].to_numpy()
    pct = df['pct_change'].to_numpy()
    last = df['is_last'].to_numpy()
    n = len(df)

    in_trade = False
    trades = []
    gain = 0.0
    n_trades = 0
    forced = 0
    n_ambiguous = 0
    n_gap = 0

    in_price = n_stocks = take_profit = stop_loss = np.nan

    for i in range(n):

        if not in_trade:
            # Signal is read on the close of bar i and filled on the open of
            # bar i+1, so we need bar i+1 to exist and to be in this session.
            if (np.isnan(pct[i])
                    or pct[i] < in_cond
                    or last[i]
                    or i + 1 >= n
                    or last[i + 1]):
                continue

            entry_bar = i + 1
            in_price = op[entry_bar]
            n_stocks = amount / in_price
            take_profit = in_price * out_up_cond
            stop_loss = in_price * out_down_cond
            in_trade = True
            # Fall through: bar i+1 is evaluated for an exit on the next pass.

        else:
            out_price = None

            if last[i]:
                # Forced flat. `is_last` is known before the bar trades (the
                # calendar tells you this is the final bar), so a market order
                # at this bar's OPEN is executable. Checked BEFORE the barriers
                # precisely because nothing inside this bar has happened yet at
                # the moment the decision is made.
                out_price = op[i]
                forced += 1

            elif op[i] >= take_profit or op[i] <= stop_loss:
                # The bar OPENED past a barrier: the market skipped the level,
                # so a resting order fills at the open, not at the level.
                # Checked first because the open is the one price in the bar
                # that is directly observed rather than inferred from a range.
                out_price = op[i]
                n_gap += 1

            elif high[i] >= take_profit and low[i] <= stop_loss:
                # Both barriers touched inside one bar. OHLC records the
                # extremes but not their ORDER, so the two paths
                #   entry -> TP -> SL   and   entry -> SL -> TP
                # are indistinguishable here. Resolve pessimistically (assume
                # the stop hit first) to get a LOWER bound on performance.
                # Watch n_ambiguous: if it is more than a couple of percent of
                # n_trades, the bar interval is too coarse for this barrier
                # width and the result rests on an unresolvable assumption.
                out_price = stop_loss
                n_ambiguous += 1

            elif low[i] <= stop_loss:
                # Stop is a TRIGGER, not a fill guarantee -- in reality you get
                # this level or worse. Booked at the level; model the shortfall
                # through cost_bps rather than pretending the fill is clean.
                out_price = stop_loss

            elif high[i] >= take_profit:
                # Limit order: fills at this price or better, so booking the
                # level exactly is conservative.
                out_price = take_profit

            if out_price is not None:
                gross = (out_price - in_price) * n_stocks
                cost = (in_price + out_price) * n_stocks * cost_bps / 1e4
                trades.append(gross - cost)
                gain += gross - cost
                n_trades += 1
                in_trade = False

    return trades, gain, n_trades, forced, n_ambiguous, n_gap


# %%
def master(ticker, interval='5m', period='max',
           change_period=60, continuous=False,
           in_cond=0.01, out_up_cond=1.01, out_down_cond=0.99,
           amount=10000, cost_bps=0.0, plot=False):

    df = load(ticker=ticker, interval=interval, period=period, plot=plot)
    df = data_prep(df=df, interval=interval,
                   change_period=change_period, continuous=continuous)

    trades, gain, n_trades, forced, n_ambiguous, n_gap = basic_strat(
        df=df,
        in_cond=in_cond,
        out_up_cond=out_up_cond,
        out_down_cond=out_down_cond,
        amount=amount,
        cost_bps=cost_bps,
    )

    trades = np.asarray(trades, dtype=float)
    win_pct = (trades > 0).mean() if n_trades else 0.0
    avg_ret = trades.mean() / amount if n_trades else 0.0

    span = f'{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}'
    print(f'Ticker: {ticker}   ({span}, {len(df)} bars)')
    print(f'Number of trades: {n_trades}'
          f'  | forced: {forced}'
          f'  | ambiguous: {n_ambiguous}'
          f'  | gap: {n_gap}')
    print(f'Total profit with {amount}USD per trade: {gain:.2f}')
    print(f'Win percentage: {win_pct:.3f}')
    print(f'Mean return per trade: {avg_ret:+.4%}')
    if n_trades:
        # The diagnostic that matters: if `forced` dominates, the +/-1% band is
        # not what closes your trades -- the 16:00 bell is -- and you are not
        # testing the strategy you think you are.
        print(f'Forced share: {forced / n_trades:.1%}'
              f'  | ambiguous share: {n_ambiguous / n_trades:.1%}')
    print()

    return trades, gain, n_trades, forced, n_ambiguous, n_gap


# %%
if __name__ == '__main__':
    tickers = ['VOO', 'SMH', 'AAPL', 'TSM', 'NVDA', 'GLD', 'TSLA', 'MSFT']

    # Gross, to compare against the earlier close-to-close versions.
    for ticker in tickers:
        master(ticker, cost_bps=0.0)

    # Net of a 10bp round trip -- roughly what the screenshot's fills imply.
    for ticker in tickers:
        master(ticker, cost_bps=5.0)

    # Continuous instrument: no session reset, no forced exits.
    # yfinance's Bitcoin ticker is 'BTC-USD'; bare 'BTC' is a listed ETF with
    # regular sessions, which would need continuous=False instead.
    master('BTC-USD', continuous=True, cost_bps=5.0)