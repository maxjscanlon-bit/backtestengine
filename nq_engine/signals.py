"""Hypothesis definitions. Each signal function takes an OHLCV frame and params,
returns a position series (+1/0/-1) using only info through each bar close.

Add new hypotheses here. Keep them small and named.
"""

import numpy as np
import pandas as pd


def zscore_reversal(df, lookback=48, z_entry=-1.5, hold_bars=6, session_rth_only=True):
    """Intraday analog of the YM strong-down reversal. Long when bar-close z-score
    of returns over `lookback` bars drops below z_entry, hold for hold_bars, exit.
    """
    # POINT returns, not pct_change. Prices are Panama back-adjusted, so early
    # history sits at an inflated level and percentage moves are systematically
    # compressed there. Point moves are unaffected by the additive offset.
    ret = df["close"].diff()
    mu = ret.rolling(lookback).mean()
    sd = ret.rolling(lookback).std()
    z = (ret - mu) / sd
    entries = (z < z_entry)
    if session_rth_only:
        t = df.index
        rth = (t.hour * 60 + t.minute >= 9 * 60 + 30) & (t.hour * 60 + t.minute < 16 * 60)
        entries = entries & rth
    live = 0
    e = np.asarray(entries.values)
    p = np.zeros(len(df))
    for i in range(len(p)):
        if e[i]:
            live = hold_bars
        if live > 0:
            p[i] = 1.0
            live -= 1
    return pd.Series(p, index=df.index)


def markov_state(df, lookback=1):
    """Placeholder for Markov transition conditioning on real returns.
    To be designed on TRAIN data only.
    """
    raise NotImplementedError
