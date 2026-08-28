"""Gartley harmonic, D inside the XA leg so the stop can anchor at X.

Bullish: pivots X(low) A(high) B(low) C(high), entry at D(low)
    AB retraces b_ret of XA        classic 0.618
    BC retraces c_lo..c_hi of AB   classic 0.382-0.886
    D  retraces d_ret of XA        classic 0.786   -> D sits ABOVE X
Bearish mirrors exactly.

Unlike the Butterfly, D does NOT overshoot X, so X is real structure below the
entry and the stop has something to sit behind. Two stop modes:
    'struct'  SL beyond X by sl_buf * |X - D|      (structural)
    'atr'     SL at atr_sl * ATR from entry        (volatility scaled)

Fills are the house conventions: limit entry requires trade-through, stop
checked before target, no target on the entry bar, gapped levels fill at the
bar open. MAE/MFE tracked. Point arithmetic only.
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


def run_gartley(df, period=10, b_ret=0.618, b_tol=0.08,
                c_lo=0.382, c_hi=0.886, d_ret=0.786, d_tol=0.05,
                stop_mode="atr", atr_n=96, atr_sl=1.5, atr_tp=1.0,
                sl_buf=0.30, tp_fib=0.618,
                friction_ticks=2.0, side="both", max_bars_pending=96,
                invert=False):
    """invert: trade the OPPOSITE direction of the pattern signal. Brackets are
    rebuilt around the flipped direction, so this is not a pure mirror: the
    entry price is the same D level but SL/TP swap sides at their own distances.
    Friction is charged either way, so inverting a loser recovers gross edge
    minus two friction payments, not the full loss."""
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

        # ---------- pattern on last four confirmed pivots ----------
        if len(zz_v) >= 4 and active is None:
            X, A, B, C = zz_v[-4], zz_v[-3], zz_v[-2], zz_v[-1]
            dX, dA, dB, dC = zz_d[-4], zz_d[-3], zz_d[-2], zz_d[-1]
            xa = A - X
            bull = dX == -1 and dA == 1 and dB == -1 and dC == 1 and xa > 0
            bear = dX == 1 and dA == -1 and dB == 1 and dC == -1 and xa < 0
            if bull or bear:
                ab = (A - B) / xa                  # AB retracement of XA
                den = (A - B)
                bc = (C - B) / den if den != 0 else np.nan
                ok_b = abs(ab - b_ret) <= b_tol
                ok_c = c_lo <= bc <= c_hi
                want = (side == "both" or (side == "long" and bull)
                        or (side == "short" and bear))
                if ok_b and ok_c and want:
                    D = A - d_ret * xa             # 78.6% retracement, inside XA
                    d = (1 if bull else -1) * (-1 if invert else 1)
                    inside = (D > X) if bull else (D < X)
                    approach = 1 if bull else -1
                    if inside:
                        a = atr[i]
                        if stop_mode == "atr":
                            SL = D - d * atr_sl * a
                            TP = D + d * atr_tp * a
                        else:
                            SL = X - d * sl_buf * abs(X - D)
                            TP = D + d * tp_fib * abs(A - D)
                        valid = (SL < D < TP) if d == 1 else (SL > D > TP)
                        if valid:
                            sig = (round(X, 4), round(A, 4), round(B, 4), round(C, 4))
                            if sig != last_sig:
                                n_patterns += 1
                                last_sig = sig
                                pending = {"entry": D, "dir": d, "approach": approach,
                                           "sl": SL, "tp": TP,
                                           "X": X, "A": A, "B": B, "C": C,
                                           "ab": ab, "bc": bc, "born": i}

        # ---------- pending limit entry ----------
        if pending is not None and active is None:
            if i - pending["born"] > max_bars_pending:
                pending = None
            else:
                d = pending["dir"]
                ap = pending["approach"]
                # invalidate if the stop zone is reached before entry
                if (d == 1 and low[i] < pending["sl"]) or (d == -1 and high[i] > pending["sl"]):
                    pending = None
                else:
                    # price approaches D from the pattern's side regardless of trade dir
                    hit = (low[i] < pending["entry"]) if ap == 1 else (high[i] > pending["entry"])
                    if hit:
                        active = {**pending, "entry_i": i, "entered_bar": i,
                                  "mae": 0.0, "mfe": 0.0}
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
                    "X": t["X"], "A": t["A"], "B": t["B"], "C": t["C"],
                    "ab_ret": round(t["ab"], 4), "bc_ret": round(t["bc"], 4),
                    "result": result, "net_points": pts, "net_pnl": dollars,
                    "bars_held": i - t["entry_i"],
                    "mae_pts": round(t["mae"], 4), "mfe_pts": round(t["mfe"], 4)})
                active = None

    pnl = pd.Series(pnl, index=idx)
    from .engine import summarize
    eq = pnl.cumsum()
    tdf = pd.DataFrame(trades)
    return {"stats": summarize(pnl, eq, tdf), "pnl": pnl, "equity": eq,
            "trades": tdf, "n_patterns": n_patterns}
