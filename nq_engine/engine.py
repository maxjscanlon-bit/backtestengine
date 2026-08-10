"""Vectorized backtest core for NQ 5-min bars.

Contract math: NQ point value $20, tick 0.25 ($5).
Signals are position series aligned to bars: +1 long, -1 short, 0 flat.
Convention: signal computed on bar t uses only data through bar t close,
position is held from bar t close to bar t+1 close (next-close execution).
That is the honest baseline. No lookahead by construction.
"""

import numpy as np
import pandas as pd

from .sessions import session_group

POINT_VALUE = 20.0
TICK = 0.25


def backtest(df, signal, friction_ticks=2.0, point_value=POINT_VALUE):
    """Run a vectorized backtest.

    df: OHLCV frame
    signal: pd.Series of desired position (+1/0/-1), index-aligned with df,
            using info through each bar close only
    friction_ticks: total ticks charged per side (spread + slippage + commission equiv)

    Returns dict with stats and the trade-level frame.
    """
    sig = signal.reindex(df.index).fillna(0.0)
    close = df["close"]
    # position held over the NEXT bar
    pos = sig.shift(1).fillna(0.0)
    bar_ret_pts = close.diff().fillna(0.0)
    gross_pts = pos * bar_ret_pts
    # friction charged on position changes (per side)
    turns = sig.diff().abs().fillna(sig.abs())
    friction_pts = turns * friction_ticks * TICK
    net_pts = gross_pts - friction_pts.shift(1).fillna(0.0)
    pnl = net_pts * point_value

    equity = pnl.cumsum()
    trades = _extract_trades(sig, close, friction_ticks, point_value)

    stats = summarize(pnl, equity, trades)
    return {"stats": stats, "pnl": pnl, "equity": equity, "trades": trades}


def backtest_subset(df, signal, idx, friction_ticks=2.0, point_value=POINT_VALUE):
    """Score a signal on a subset of bars WITHOUT slicing the price series.

    The signal and the bar returns are computed on the full contiguous frame,
    then P&L is accrued only on bars in `idx`. The first bar of each contiguous
    block in `idx` is dropped because its return spans a gap to whatever came
    before the block, which is not a real market move.

    This is the correct way to evaluate CPCV folds. Slicing the frame first and
    recomputing rolling features across the seam is what produced fake jumps.
    """
    sig = signal.reindex(df.index).fillna(0.0)
    pos = sig.shift(1).fillna(0.0)
    bar_ret_pts = df["close"].diff().fillna(0.0)
    friction_pts = sig.diff().abs().fillna(sig.abs()) * friction_ticks * TICK
    net_pts = pos * bar_ret_pts - friction_pts.shift(1).fillna(0.0)
    pnl_full = net_pts * point_value

    idx = np.asarray(idx)
    keep = np.ones(len(idx), dtype=bool)
    keep[0] = False
    keep[1:] = np.diff(idx) == 1          # drop first bar of every block
    sel = idx[keep]

    pnl = pnl_full.iloc[sel]
    equity = pnl.cumsum()
    sub_sig = sig.iloc[sel]
    trades = _extract_trades(sub_sig, df["close"].iloc[sel], friction_ticks, point_value)
    return {"stats": summarize(pnl, equity, trades), "pnl": pnl,
            "equity": equity, "trades": trades}


def _extract_trades(sig, close, friction_ticks, point_value):
    """Collapse the position series into discrete trades."""
    rows = []
    in_pos = 0.0
    entry_px = np.nan
    entry_t = None
    s = sig.values
    c = close.values
    idx = sig.index
    for i in range(len(s)):
        cur = s[i]
        if cur != in_pos:
            if in_pos != 0.0:
                pts = (c[i] - entry_px) * in_pos
                cost = 2 * friction_ticks * TICK
                rows.append({
                    "entry_time": entry_t, "exit_time": idx[i],
                    "dir": in_pos, "entry": entry_px, "exit": c[i],
                    "points": pts, "net_points": pts - cost,
                    "net_pnl": (pts - cost) * point_value,
                })
            if cur != 0.0:
                entry_px = c[i]
                entry_t = idx[i]
            in_pos = cur
    return pd.DataFrame(rows)


def summarize(pnl, equity, trades):
    stats = {}
    stats["total_pnl"] = round(float(pnl.sum()), 2)
    stats["n_trades"] = len(trades)
    if len(trades):
        stats["expectancy_per_trade"] = round(float(trades["net_pnl"].mean()), 2)
        stats["win_rate"] = round(float((trades["net_pnl"] > 0).mean()), 4)
        gp = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
        gl = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
        stats["profit_factor"] = round(float(gp / gl), 3) if gl > 0 else float("inf")
    dd = equity - equity.cummax()
    stats["max_drawdown"] = round(float(dd.min()), 2)
    daily = session_group(pnl).sum()
    if daily.std() > 0:
        stats["daily_sharpe_ann"] = round(float(daily.mean() / daily.std() * np.sqrt(252)), 2)
    return stats


def friction_stress(df, signal, ticks_grid=(0, 1, 2, 3, 4)):
    """Rerun the same signal across a friction grid. Edge must survive realistic ticks."""
    rows = []
    for t in ticks_grid:
        r = backtest(df, signal, friction_ticks=t)
        row = {"friction_ticks": t}
        row.update(r["stats"])
        rows.append(row)
    return pd.DataFrame(rows)
