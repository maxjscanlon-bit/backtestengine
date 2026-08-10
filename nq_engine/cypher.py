"""Cypher harmonic pattern on confirmed ZigZag pivots.

Bullish: pivots X(low) A(high) B(low) C(high) with
    B retracing rb in [0.382, 0.618] of XA
    C extending rc in [1.272, 1.414] of XA measured from X
Entry: buy limit at D = C - 0.786 * (C - X)   (78.6% retracement of XC)
SL: below X by a small buffer. TP: tp_frac retracement of CD back toward C.
Bearish is the exact mirror.

Pivot logic is identical to mw.py (period-based ZigZag, confirmed pivots only).
Fills are conservative: trade-through entries, SL checked first, no TP on the
entry bar. One position at a time; a newer valid pattern replaces a pending
(inactive) order. All math in points, Panama-immune.
"""

import numpy as np
import pandas as pd

POINT_VALUE = 20.0
TICK = 0.25


def run_cypher(df, period=10, b_lo=0.382, b_hi=0.618, c_lo=1.272, c_hi=1.414,
               d_ret=0.786, tp_frac=0.382, sl_buffer_frac=0.1,
               friction_ticks=2.0):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)
    idx = df.index

    hh = pd.Series(high, index=idx).rolling(period - 1).max().shift(1).values
    ll = pd.Series(low, index=idx).rolling(period - 1).min().shift(1).values

    pnl = np.zeros(n)
    trades = []
    zz_vals, zz_dirs = [], []
    cur_val, cur_dir, dir_ = None, 0, 0
    pending, active = None, None
    fric_pts = friction_ticks * TICK

    for i in range(n):
        if i < period or np.isnan(hh[i]):
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

        # cypher check on last four confirmed pivots
        if len(zz_vals) >= 4 and active is None:
            X, A, B, C = zz_vals[-4], zz_vals[-3], zz_vals[-2], zz_vals[-1]
            dX, dA, dB, dC = zz_dirs[-4], zz_dirs[-3], zz_dirs[-2], zz_dirs[-1]
            xa = A - X
            bull = (dX == -1 and dA == 1 and dB == -1 and dC == 1 and xa > 0)
            bear = (dX == 1 and dA == -1 and dB == 1 and dC == -1 and xa < 0)
            if bull or bear:
                rb = (A - B) / xa            # B retracement of XA
                rc = (C - X) / xa            # C extension of XA from X
                if b_lo <= rb <= b_hi and c_lo <= rc <= c_hi:
                    D = C - d_ret * (C - X)
                    slb = abs(C - X) * sl_buffer_frac
                    if bull:
                        sl = X - slb
                        tp = D + tp_frac * (C - D)
                        pending = {"entry": D, "dir": 1, "tp": tp, "sl": sl,
                                   "X": X, "A": A, "B": B, "C": C}
                    else:
                        sl = X + slb
                        tp = D - tp_frac * (D - C)
                        pending = {"entry": D, "dir": -1, "tp": tp, "sl": sl,
                                   "X": X, "A": A, "B": B, "C": C}

        # invalidate pending if price passes the stop zone before entry
        if pending is not None and active is None:
            if (pending["dir"] == 1 and low[i] < pending["sl"]) or \
               (pending["dir"] == -1 and high[i] > pending["sl"]):
                pending = None

        entered_this_bar = False
        if active is None and pending is not None:
            if pending["dir"] == 1 and low[i] < pending["entry"]:
                active = dict(pending, entry_i=i)
                pending = None
                entered_this_bar = True
            elif pending["dir"] == -1 and high[i] > pending["entry"]:
                active = dict(pending, entry_i=i)
                pending = None
                entered_this_bar = True

        if active is not None:
            exit_px, result = None, 0
            op = open_[i]
            if active["dir"] == 1:
                if low[i] <= active["sl"]:
                    exit_px = min(op, active["sl"])   # gap through stop fills worse
                    result = -1
                elif not entered_this_bar and high[i] >= active["tp"]:
                    exit_px = max(op, active["tp"])   # gap through target fills better
                    result = 1
            else:
                if high[i] >= active["sl"]:
                    exit_px = max(op, active["sl"])
                    result = -1
                elif not entered_this_bar and low[i] <= active["tp"]:
                    exit_px = min(op, active["tp"])
                    result = 1
            if exit_px is not None:
                pts = (exit_px - active["entry"]) * active["dir"] - 2 * fric_pts
                dollars = pts * POINT_VALUE
                pnl[i] += dollars
                trades.append({
                    "entry_time": idx[active["entry_i"]], "exit_time": idx[i],
                    "dir": active["dir"],
                    "X": active["X"], "A": active["A"], "B": active["B"],
                    "C": active["C"],
                    "entry": active["entry"], "exit": exit_px,
                    "tp_level": active["tp"], "sl_level": active["sl"],
                    "result": result, "net_points": pts, "net_pnl": dollars,
                    "bars_held": i - active["entry_i"],
                })
                active = None

    pnl = pd.Series(pnl, index=idx)
    trades = pd.DataFrame(trades)
    from .engine import summarize
    equity = pnl.cumsum()
    return {"stats": summarize(pnl, equity, trades), "pnl": pnl,
            "equity": equity, "trades": trades}
