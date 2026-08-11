"""Cypher with configurable concurrent position limit.

The original engine allowed one position at a time, which was an arbitrary
restriction, not a real constraint. This version tracks a list of open trades
and a list of resting orders, so max_concurrent controls how many patterns can
be live at once. max_concurrent=1 reproduces the original behaviour.

Everything else is identical: ATR brackets, conservative fills, gap handling.
"""

import numpy as np
import pandas as pd

POINT_VALUE = 20.0
TICK = 0.25


def _atr(df, n=96):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().shift(1)


def run_cypher_multi(df, period=10, b_lo=0.382, b_hi=0.618, c_lo=1.272,
                     c_hi=1.414, d_ret=0.786, atr_n=96, atr_sl=1.5, atr_tp=1.0,
                     friction_ticks=2.0, side="both", max_concurrent=1,
                     max_resting=1):
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)
    idx = df.index
    atr = _atr(df, atr_n).values

    hh = pd.Series(high, index=idx).rolling(period - 1).max().shift(1).values
    ll = pd.Series(low, index=idx).rolling(period - 1).min().shift(1).values

    pnl = np.zeros(n)
    trades = []
    zz_vals, zz_dirs = [], []
    cur_val, cur_dir, dir_ = None, 0, 0
    resting = []      # list of dicts
    active = []       # list of dicts
    fric = friction_ticks * TICK
    peak_open = 0

    for i in range(n):
        if i < period or np.isnan(hh[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        has_ph = high[i] > hh[i]
        has_pl = low[i] < ll[i]
        if has_ph and not has_pl:
            dir_ = 1
        elif has_pl and not has_ph:
            dir_ = -1
        if cur_val is not None:
            if cur_dir == 1 and has_ph and high[i] > cur_val:
                cur_val = high[i]
            elif cur_dir == -1 and has_pl and low[i] < cur_val:
                cur_val = low[i]
        if has_ph or has_pl:
            if cur_val is not None and dir_ != cur_dir:
                zz_vals.append(cur_val)
                zz_dirs.append(cur_dir)
                cur_val = None
            if cur_val is None:
                cur_val = high[i] if dir_ == 1 else low[i]
                cur_dir = dir_

        # ---- new pattern -> add a resting order if capacity allows
        if len(zz_vals) >= 4 and len(active) < max_concurrent and len(resting) < max_resting:
            X, A, B, C = zz_vals[-4], zz_vals[-3], zz_vals[-2], zz_vals[-1]
            dX, dA, dB, dC = zz_dirs[-4], zz_dirs[-3], zz_dirs[-2], zz_dirs[-1]
            xa = A - X
            bull = dX == -1 and dA == 1 and dB == -1 and dC == 1 and xa > 0
            bear = dX == 1 and dA == -1 and dB == 1 and dC == -1 and xa < 0
            if bull or bear:
                rb = (A - B) / xa
                rc = (C - X) / xa
                want = (side == "both" or (side == "long" and bull)
                        or (side == "short" and bear))
                if b_lo <= rb <= b_hi and c_lo <= rc <= c_hi and want:
                    D = C - d_ret * (C - X)
                    a = atr[i]
                    d = 1 if bull else -1
                    sl = D - d * atr_sl * a
                    tp = D + d * atr_tp * a
                    key = (round(X, 4), round(A, 4), round(B, 4), round(C, 4))
                    if not any(r["key"] == key for r in resting) and \
                       not any(t["key"] == key for t in active):
                        resting.append({"entry": D, "dir": d, "tp": tp, "sl": sl,
                                        "key": key})

        # ---- invalidate resting orders whose stop zone is breached
        resting = [r for r in resting
                   if not ((r["dir"] == 1 and low[i] < r["sl"]) or
                           (r["dir"] == -1 and high[i] > r["sl"]))]

        # ---- activate resting orders on trade-through
        still = []
        for r in resting:
            hit = (r["dir"] == 1 and low[i] < r["entry"]) or \
                  (r["dir"] == -1 and high[i] > r["entry"])
            if hit and len(active) < max_concurrent:
                active.append({**r, "entry_i": i, "entered_bar": i,
                               "mae": 0.0, "mfe": 0.0})
            else:
                still.append(r)
        resting = still
        peak_open = max(peak_open, len(active))

        # ---- manage open trades
        keep = []
        for t in active:
            # excursions in points, favourable/adverse relative to entry
            up = (high[i] - t["entry"]) * t["dir"]
            dn = (low[i] - t["entry"]) * t["dir"]
            t["mfe"] = max(t["mfe"], up if t["dir"] == 1 else -dn * -1)
            t["mae"] = min(t["mae"], dn if t["dir"] == 1 else -up * -1)
            if t["dir"] == -1:
                t["mfe"] = max(t["mfe"], (t["entry"] - low[i]))
                t["mae"] = min(t["mae"], (t["entry"] - high[i]))
            exit_px, result = None, 0
            entered_this_bar = t["entered_bar"] == i
            op = open_[i]
            if t["dir"] == 1:
                if low[i] <= t["sl"]:
                    exit_px, result = min(op, t["sl"]), -1
                elif not entered_this_bar and high[i] >= t["tp"]:
                    exit_px, result = max(op, t["tp"]), 1
            else:
                if high[i] >= t["sl"]:
                    exit_px, result = max(op, t["sl"]), -1
                elif not entered_this_bar and low[i] <= t["tp"]:
                    exit_px, result = min(op, t["tp"]), 1
            if exit_px is None:
                keep.append(t)
            else:
                pts = (exit_px - t["entry"]) * t["dir"] - 2 * fric
                dollars = pts * POINT_VALUE
                pnl[i] += dollars
                trades.append({"entry_time": idx[t["entry_i"]], "exit_time": idx[i],
                               "dir": t["dir"], "entry": t["entry"], "exit": exit_px,
                               "tp_level": t["tp"], "sl_level": t["sl"],
                               "result": result, "net_points": pts,
                               "net_pnl": dollars, "bars_held": i - t["entry_i"],
                               "mae_pts": round(t["mae"], 4),
                               "mfe_pts": round(t["mfe"], 4)})
        active = keep

    pnl = pd.Series(pnl, index=idx)
    from .engine import summarize
    eq = pnl.cumsum()
    tdf = pd.DataFrame(trades)
    return {"stats": summarize(pnl, eq, tdf), "pnl": pnl, "equity": eq,
            "trades": tdf, "peak_concurrent": peak_open}
