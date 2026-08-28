"""Regime classification. All features shifted one bar so they are known
before the bar they label. Point-based, Panama-immune."""
import numpy as np
import pandas as pd


def _atr(df, n=96):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().shift(1)


def regimes(df, atr_n=96, er_n=96, lookback=5520, lo=0.33, hi=0.67):
    atr = _atr(df, atr_n)
    vol_rank = atr.rolling(lookback, min_periods=1000).rank(pct=True)
    net = (df["close"] - df["close"].shift(er_n)).abs()
    tot = df["close"].diff().abs().rolling(er_n).sum()
    er = (net / tot).shift(1)
    er_rank = er.rolling(lookback, min_periods=1000).rank(pct=True)

    def bucket(x, labels):
        return pd.cut(x, [-0.01, lo, hi, 1.01], labels=labels)

    # DIRECTIONAL regime: sign and strength of longer-term drift, in points,
    # measured over dir_n bars and shifted one bar so it is known in advance.
    dir_n = 1560                      # ~1 week of 5-min bars
    drift = (df["close"] - df["close"].shift(dir_n)).shift(1)
    drift_atr = drift / atr.replace(0, np.nan)
    dir_state = pd.cut(drift_atr, [-1e9, -1.0, 1.0, 1e9],
                       labels=["bear", "neutral", "bull"])

    return pd.DataFrame({
        "atr": atr, "vol_pct": vol_rank, "er": er, "er_pct": er_rank,
        "vol_state": bucket(vol_rank, ["low", "mid", "high"]),
        "trend_state": bucket(er_rank, ["chop", "mixed", "trend"]),
        "drift_atr": drift_atr, "dir_state": dir_state,
    }, index=df.index)


def condition_trades(trades, reg, on="entry_time"):
    t = trades.copy()
    idx = pd.DatetimeIndex(t[on])
    for col in ("vol_state", "trend_state", "dir_state", "vol_pct", "er_pct", "drift_atr"):
        t[col] = reg[col].reindex(idx).values
    return t
