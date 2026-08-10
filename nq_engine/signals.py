"""Hypothesis definitions. Each signal function takes an OHLCV frame and params,
returns a position series (+1/0/-1) using only info through each bar close.

Add new hypotheses here. Keep them small and named.
"""

import numpy as np
import pandas as pd

from .sessions import rth_mask


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
        entries = entries & rth_mask(df.index)
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


def zscore_twosided(df, lookback=48, z_entry=1.5, hold_bars=6, mode="reversal",
                    session_rth_only=True):
    """Symmetric z-score strategy on point returns.

    mode='reversal': long when z < -z_entry, short when z > +z_entry
    mode='momentum': the mirror, long on strong up, short on strong down
    Position held for hold_bars bars after each trigger; retriggering extends.
    """
    ret = df["close"].diff()
    mu = ret.rolling(lookback).mean()
    sd = ret.rolling(lookback).std()
    z = (ret - mu) / sd
    long_trig = (z < -z_entry).values
    short_trig = (z > z_entry).values
    if mode == "momentum":
        long_trig, short_trig = short_trig, long_trig
    if session_rth_only:
        m = rth_mask(df.index)
        long_trig = long_trig & m
        short_trig = short_trig & m
    p = np.zeros(len(df))
    live, side = 0, 0.0
    for i in range(len(p)):
        if long_trig[i]:
            live, side = hold_bars, 1.0
        elif short_trig[i]:
            live, side = hold_bars, -1.0
        if live > 0:
            p[i] = side
            live -= 1
    return pd.Series(p, index=df.index)


def markov_state(df, lookback=1):
    """Placeholder for Markov transition conditioning on real returns.
    To be designed on TRAIN data only.
    """
    raise NotImplementedError
