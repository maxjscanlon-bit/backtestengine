"""Double Bottom / Double Top, neckline-break confirmation entry.

Three confirmed ZigZag pivots. Double bottom: L1(low) N(high) L2(low)
    |L1 - L2| <= equal_tol * depth      where depth = N - min(L1, L2)
    depth >= depth_min * ATR            pattern must be meaningful vs current vol
    optional L2 >= L1 (higher low) via require_higher_low
Neckline = N. Entry is a BUY STOP above N, filled on trade-through.
Double top mirrors exactly.

Distinct from the M/W family already tested: those entered on a LIMIT at the
pivot retest (fade). This enters on a STOP through the neckline (confirmation),
which is the textbook double-bottom trigger.

Engine constraints applied throughout:
  - ATR(96) brackets, shifted one bar (never fixed geometry, never pattern Fibs)
  - point arithmetic only (Panama-immune)
  - entries require trade-through, not touch
  - stop-loss checked before target
  - no target on the entry bar
  - levels gapped through fill at the bar open
  - MAE/MFE tracked per trade
  - pattern signature dedupe so one setup arms once
"""

import numpy as np
import pandas as pd

POINT_VALUE = 20.0
TICK = 0.25


def entered_this_bar_flag(t, i):
    return t["entered_bar"] == i


def _atr(df, n=96):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().shift(1)


def run_double(df, period=10, equal_tol=0.15, depth_min=0.75,
               require_higher_low=False, atr_n=96, atr_sl=3.0, atr_tp=1.5,
               friction_ticks=2.0, side="both", max_bars_pending=48,
               trail_atr=None, be_trigger=None):
    """trail_atr : if set, stop trails the best close by trail_atr * ATR
       be_trigger: if set, stop moves to entry once MFE >= be_trigger * ATR
       Both are checked only on bars AFTER entry and never loosen the stop."""
    """side: 'both' | 'bottom' (long only) | 'top' (short only)"""
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
    zz_v, zz_d = [], []
    cur_val, cur_dir, dir_ = None, 0, 0
    pending, active = None, None
    fric = friction_ticks * TICK
    n_patterns = 0
    last_sig = None

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
                zz_v.append(cur_val)
                zz_d.append(cur_dir)
                cur_val = None
            if cur_val is None:
                cur_val = high[i] if dir_ == 1 else low[i]
                cur_dir = dir_

        # ---------- pattern detection on last three confirmed pivots ----------
        if len(zz_v) >= 3 and active is None:
            P1, N, P2 = zz_v[-3], zz_v[-2], zz_v[-1]
            d1, dN, d2 = zz_d[-3], zz_d[-2], zz_d[-1]
            a = atr[i]

            bottom = (d1 == -1 and dN == 1 and d2 == -1)
            top = (d1 == 1 and dN == -1 and d2 == 1)

            ok, d = False, 0
            if bottom:
                depth = N - min(P1, P2)
                if depth > 0:
                    ok = (abs(P1 - P2) <= equal_tol * depth
                          and depth >= depth_min * a
                          and (not require_higher_low or P2 >= P1))
                d = 1
            elif top:
                depth = max(P1, P2) - N
                if depth > 0:
                    ok = (abs(P1 - P2) <= equal_tol * depth
                          and depth >= depth_min * a
                          and (not require_higher_low or P2 <= P1))
                d = -1

            want = (side == "both"
                    or (side == "bottom" and bottom)
                    or (side == "top" and top))
            if ok and want:
                sig = (round(P1, 4), round(N, 4), round(P2, 4))
                if sig != last_sig:
                    n_patterns += 1
                    last_sig = sig
                    pending = {"neck": N, "dir": d, "born": i,
                               "P1": P1, "P2": P2,
                               "depth": (N - min(P1, P2)) if bottom
                                        else (max(P1, P2) - N)}

        # ---------- pending: stop entry on neckline break ----------
        if pending is not None and active is None:
            if i - pending["born"] > max_bars_pending:
                pending = None
            else:
                neck = pending["neck"]
                d = pending["dir"]
                hit = (high[i] > neck) if d == 1 else (low[i] < neck)
                if hit:
                    op = open_[i]
                    entry = max(op, neck) if d == 1 else min(op, neck)
                    a = atr[i]
                    active = {"entry": entry, "dir": d,
                              "sl": entry - d * atr_sl * a,
                              "tp": entry + d * atr_tp * a,
                              "entry_i": i, "entered_bar": i,
                              "mae": 0.0, "mfe": 0.0,
                              "P1": pending["P1"], "P2": pending["P2"],
                              "neck": neck, "depth": pending["depth"],
                              "atr0": a, "be_done": False}
                    pending = None

        # ---------- manage open trade ----------
        if active is not None:
            t = active
            if t["dir"] == 1:
                t["mfe"] = max(t["mfe"], high[i] - t["entry"])
                t["mae"] = min(t["mae"], low[i] - t["entry"])
            else:
                t["mfe"] = max(t["mfe"], t["entry"] - low[i])
                t["mae"] = min(t["mae"], t["entry"] - high[i])

            # ---- trailing / breakeven stop management (never loosens) ----
            if not entered_this_bar_flag(t, i):
                a0 = t["atr0"]
                if be_trigger is not None and not t["be_done"] and t["mfe"] >= be_trigger * a0:
                    t["sl"] = max(t["sl"], t["entry"]) if t["dir"] == 1 else min(t["sl"], t["entry"])
                    t["be_done"] = True
                if trail_atr is not None:
                    if t["dir"] == 1:
                        t["sl"] = max(t["sl"], t["entry"] + t["mfe"] - trail_atr * a0)
                    else:
                        t["sl"] = min(t["sl"], t["entry"] - t["mfe"] + trail_atr * a0)

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

            if exit_px is not None:
                pts = (exit_px - t["entry"]) * t["dir"] - 2 * fric
                dollars = pts * POINT_VALUE
                pnl[i] += dollars
                trades.append({
                    "entry_time": idx[t["entry_i"]], "exit_time": idx[i],
                    "dir": t["dir"], "entry": t["entry"], "exit": exit_px,
                    "tp_level": t["tp"], "sl_level": t["sl"],
                    "P1": t["P1"], "P2": t["P2"], "neck": t["neck"],
                    "depth": round(t["depth"], 4),
                    "result": result, "net_points": pts, "net_pnl": dollars,
                    "bars_held": i - t["entry_i"],
                    "mae_pts": round(t["mae"], 4),
                    "mfe_pts": round(t["mfe"], 4)})
                active = None

    pnl = pd.Series(pnl, index=idx)
    from .engine import summarize
    eq = pnl.cumsum()
    tdf = pd.DataFrame(trades)
    return {"stats": summarize(pnl, eq, tdf), "pnl": pnl, "equity": eq,
            "trades": tdf, "n_patterns": n_patterns}
