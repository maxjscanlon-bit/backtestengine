"""Head & Shoulders and Inverse H&S, neckline-break breakout.

Five confirmed ZigZag pivots. Bearish H&S: LS(high) T1(low) H(high) T2(low) RS(high)
    H > LS, H > RS, |LS - RS| <= shoulder_tol * head_prominence
    head_prominence = H - max(LS, RS)  must be >= head_min * ATR
Neckline runs through T1 and T2; entry is a SELL STOP at the neckline price
projected to the current bar. Inverse H&S mirrors everything.

Unlike the harmonic patterns this is a BREAKOUT: the entry stop is placed
beyond the neckline and fills when price trades through it, so a valid pattern
that never breaks produces no trade.

Fill conventions match cypher_multi exactly:
  - stop entries require trade-through (strictly beyond the level)
  - stop-loss checked before target
  - no target on the entry bar
  - levels gapped through fill at the bar open
Brackets are ATR(96) multiples, shifted one bar. MAE/MFE tracked.
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


def run_hs(df, period=10, shoulder_tol=0.35, head_min=0.5,
           neckline_slope_max=0.5, atr_n=96, atr_sl=1.5, atr_tp=1.0,
           friction_ticks=2.0, side="both", max_bars_pending=48,
           max_concurrent=1):
    """side: 'both' | 'bear' (classic H&S, short) | 'bull' (inverse, long)"""
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
    zz_v, zz_d, zz_i = [], [], []          # value, direction, bar index
    cur_val, cur_dir, cur_i, dir_ = None, 0, 0, 0
    pending, active = None, []
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
                cur_val, cur_i = high[i], i
            elif cur_dir == -1 and has_pl and low[i] < cur_val:
                cur_val, cur_i = low[i], i

        if has_ph or has_pl:
            if cur_val is not None and dir_ != cur_dir:
                zz_v.append(cur_val); zz_d.append(cur_dir); zz_i.append(cur_i)
                cur_val = None
            if cur_val is None:
                cur_val = high[i] if dir_ == 1 else low[i]
                cur_dir, cur_i = dir_, i

        # ---------- pattern detection on last five confirmed pivots ----------
        if len(zz_v) >= 5 and not active:
            LS, T1, H, T2, RS = zz_v[-5:]
            dLS, dT1, dH, dT2, dRS = zz_d[-5:]
            iT1, iT2 = zz_i[-4], zz_i[-2]
            a = atr[i]

            bear = (dLS == 1 and dT1 == -1 and dH == 1 and dT2 == -1 and dRS == 1
                    and H > LS and H > RS)
            bull = (dLS == -1 and dT1 == 1 and dH == -1 and dT2 == 1 and dRS == -1
                    and H < LS and H < RS)

            ok = False
            if bear:
                prom = H - max(LS, RS)
                sym = abs(LS - RS)
                ok = (prom >= head_min * a and sym <= shoulder_tol * prom)
                d = -1
            elif bull:
                prom = min(LS, RS) - H
                sym = abs(LS - RS)
                ok = (prom >= head_min * a and sym <= shoulder_tol * prom)
                d = 1

            if ok:
                span = max(iT2 - iT1, 1)
                slope = (T2 - T1) / span               # points per bar
                if abs(slope) <= neckline_slope_max * a / max(span, 1) * span:
                    want = (side == "both"
                            or (side == "bear" and bear)
                            or (side == "bull" and bull))
                    sig = (round(LS, 4), round(H, 4), round(RS, 4))
                    if want and sig != last_sig:
                        n_patterns += 1
                        last_sig = sig
                        pending = {"T1": T1, "T2": T2, "iT1": iT1, "iT2": iT2,
                                   "slope": slope, "dir": d, "born": i,
                                   "LS": LS, "H": H, "RS": RS, "prom": prom}

        # ---------- pending: project neckline, arm stop entry ----------
        if pending is not None and not active:
            if i - pending["born"] > max_bars_pending:
                pending = None
            else:
                neck = pending["T2"] + pending["slope"] * (i - pending["iT2"])
                d = pending["dir"]
                # breakout: short fills BELOW neckline, long fills ABOVE
                hit = (low[i] < neck) if d == -1 else (high[i] > neck)
                if hit:
                    op = open_[i]
                    # gapped-through entries fill at the open (worse)
                    entry = min(op, neck) if d == -1 else max(op, neck)
                    a = atr[i]
                    active.append({
                        "entry": entry, "dir": d,
                        "sl": entry - d * atr_sl * a,
                        "tp": entry + d * atr_tp * a,
                        "entry_i": i, "entered_bar": i, "mae": 0.0, "mfe": 0.0,
                        "LS": pending["LS"], "H": pending["H"],
                        "RS": pending["RS"], "neck": neck,
                        "prom": pending["prom"]})
                    pending = None

        # ---------- manage open trades ----------
        keep = []
        for t in active:
            if t["dir"] == 1:
                t["mfe"] = max(t["mfe"], high[i] - t["entry"])
                t["mae"] = min(t["mae"], low[i] - t["entry"])
            else:
                t["mfe"] = max(t["mfe"], t["entry"] - low[i])
                t["mae"] = min(t["mae"], t["entry"] - high[i])

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
                trades.append({
                    "entry_time": idx[t["entry_i"]], "exit_time": idx[i],
                    "dir": t["dir"], "entry": t["entry"], "exit": exit_px,
                    "tp_level": t["tp"], "sl_level": t["sl"],
                    "LS": t["LS"], "H": t["H"], "RS": t["RS"],
                    "neck": t["neck"], "prom": t["prom"],
                    "result": result, "net_points": pts, "net_pnl": dollars,
                    "bars_held": i - t["entry_i"],
                    "mae_pts": round(t["mae"], 4),
                    "mfe_pts": round(t["mfe"], 4)})
        active = keep

    pnl = pd.Series(pnl, index=idx)
    from .engine import summarize
    eq = pnl.cumsum()
    tdf = pd.DataFrame(trades)
    return {"stats": summarize(pnl, eq, tdf), "pnl": pnl, "equity": eq,
            "trades": tdf, "n_patterns": n_patterns}
