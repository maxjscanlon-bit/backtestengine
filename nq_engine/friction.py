"""Volatility-scaled friction.

A flat ticks-per-side assumption understates cost exactly when it matters most:
spreads widen and slippage grows when volatility spikes, and pattern strategies
tend to trigger in those conditions. This models cost as

    ticks_per_side = base + slope * (ATR_now / ATR_median - 1)

clamped to [base, cap]. With slope=0 it reduces to flat friction, so existing
results remain reproducible.
"""
import numpy as np
import pandas as pd

TICK = 0.25


def vol_scaled_ticks(df, atr_n=96, base=2.0, slope=2.0, cap=6.0):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_n, adjust=False).mean().shift(1)
    ratio = atr / atr.median()
    ticks = base + slope * (ratio - 1.0)
    return ticks.clip(lower=base, upper=cap)
