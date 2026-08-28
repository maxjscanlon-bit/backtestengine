"""Consolidation-breakout continuation family: bull/bear flag, rectangle,
ascending/descending triangle share one core: impulse -> tight consolidation ->
breakout in the impulse direction.

    impulse:       |close[i] - close[i-imp_bars]| >= imp_atr * ATR
    consolidation: the next cons_bars bars stay in a range <= cons_atr * ATR
    entry:         stop order at the consolidation extreme in the impulse
                   direction, armed for max_bars_pending bars
    brackets:      ATR multiples from entry (house rule)

House fill conventions: stop entries require trade-through, stop-loss checked
first, no target on the entry bar, gapped levels fill at the open. MAE/MFE
tracked. Point arithmetic only.
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


def run_flag(df, imp_bars=12, imp_atr=4.0, cons_bars=8, cons_atr=2.0,
             atr_n=96, atr_sl=1.5, atr_tp=1.0, friction_ticks=2.0,
             side="both", max_bars_pending=24):
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    n = len(df)
    idx = df.index
    atr = _atr(df, atr_n).values

    pnl = np.zeros(n)
    trades = []
    pending, active = None, None
    fric = friction_ticks * TICK
    n_patterns = 0
    last_sig = None

    for i in range(n):
        if i < imp_bars + cons_bars + 1 or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # ---------- detect impulse -> consolidation ending at bar i ----------
        if active is None and pending is None:
            j0 = i - cons_bars + 1            # consolidation window [j0, i]
            k0 = j0 - imp_bars                # impulse window ends at j0-1
            imp = close[j0 - 1] - close[k0 - 1]
            a = atr[i]
            if abs(imp) >= imp_atr * a:
                ch = high[j0:i + 1].max()
                cl = low[j0:i + 1].min()
                if (ch - cl) <= cons_atr * a:
                    d = 1 if imp > 0 else -1
                    if side == "both" or (side == "long" and d == 1) \
                            or (side == "short" and d == -1):
                        trigger = ch if d == 1 else cl
                        sig = (round(trigger, 4), d)
                        if sig != last_sig:
                            n_patterns += 1
                            last_sig = sig
                            pending = {"trigger": trigger, "dir": d, "born": i}

        # ---------- pending stop entry on consolidation break ----------
        if pending is not None and active is None:
            if i - pending["born"] > max_bars_pending:
                pending = None
            elif i > pending["born"]:
                d = pending["dir"]
                trig = pending["trigger"]
                hit = (high[i] > trig) if d == 1 else (low[i] < trig)
                if hit:
                    op = open_[i]
                    entry = max(op, trig) if d == 1 else min(op, trig)
                    a = atr[i]
                    active = {"entry": entry, "dir": d,
                              "sl": entry - d * atr_sl * a,
                              "tp": entry + d * atr_tp * a,
                              "entry_i": i, "entered_bar": i,
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
