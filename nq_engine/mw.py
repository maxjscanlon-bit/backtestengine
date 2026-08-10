"""Port of the ZigZag M/W pattern strategy (Default mode: M long, W short).

Faithful to the NinjaScript source in signal logic, deliberately NOT faithful
in fill assumptions, where the original is optimistic:

Original pseudo-engine behaviors we do NOT replicate:
  - Activation on touch (low <= entry). We require trade-through (low < entry).
  - TP checked before SL when both sit inside one bar. We assume SL first.
  - Entry and favorable exit on the same bar. We allow only SL on the entry bar.

Behaviors we DO replicate exactly:
  - ZigZag pivot logic: hasPH = high > MAX(high, period-1)[1 bar ago], mirror
    for lows; direction flips confirm the provisional pivot into the list.
  - M = pivots (low, high, low) with l1 < h and l2 < h and l1 < l2 is NOT
    required; the source requires zz1 < zz2, zz2 > zz3, zz1 < zz3.
    (second low above first low). W is the mirror.
  - Entry limit at the third pivot's price. While the order is pending and the
    pattern remains the latest confirmed triple, the bracket is recomputed each
    bar from the current bar extreme (RefPrice drifts, entry does not).
  - TP/SL = entry +/- (|extreme - entry| * pct / 100), signs per source.
  - A new confirmed pattern replaces a pending (inactive) order.
  - One position at a time per instance.

All price math is in points, invariant to the Panama offset.
"""

import numpy as np
import pandas as pd

POINT_VALUE = 20.0
TICK = 0.25


def run_mw(df, period=5, tp_pct=61.8, sl_pct=-61.8, pattern="both",
           friction_ticks=2.0, exit_mode="bracket", hold_bars=12,
           trade_direction="default"):
    """Event-driven backtest of the Default-mode M/W strategy on OHLC bars.

    exit_mode:
      'bracket'  original TP/SL percent bracket (conservative resolution)
      'time'     no stop, no target: exit at the close hold_bars after entry
      'farstop'  stop beyond the pattern's FAR pivot (v1), TP percent bracket;
                 stop rationale: v1 is structure, a fraction of the swing is noise

    Returns {'stats', 'pnl', 'trades'} shaped like engine.backtest output.
    pnl is a per-bar series with each closed trade's P&L attributed to its
    exit bar (zero elsewhere), so fold scoring and session Sharpe work.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)
    idx = df.index

    # rolling extremes of the PRIOR period-1 bars (exclusive of current bar)
    hh = pd.Series(high, index=idx).rolling(period - 1).max().shift(1).values
    ll = pd.Series(low, index=idx).rolling(period - 1).min().shift(1).values

    pnl = np.zeros(n)
    trades = []

    zz_vals, zz_dirs = [], []       # confirmed pivots
    cur_val, cur_dir = None, 0      # provisional pivot
    dir_ = 0

    pending = None   # dict: entry, dir, ptype
    active = None    # dict: entry, tp, sl, dir, ptype, entry_i

    want_m = pattern in ("m", "both")
    want_w = pattern in ("w", "both")
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

        # extend provisional pivot in its own direction
        if cur_val is not None:
            if cur_dir == 1 and has_ph and high[i] > cur_val:
                cur_val = high[i]
            elif cur_dir == -1 and has_pl and low[i] < cur_val:
                cur_val = low[i]

        confirmed = False
        if has_ph or has_pl:
            if cur_val is not None and dir_ != cur_dir:
                zz_vals.append(cur_val)
                zz_dirs.append(cur_dir)
                confirmed = True
                cur_val = None
            if cur_val is None:
                cur_val = high[i] if dir_ == 1 else low[i]
                cur_dir = dir_

        # pattern check on the last three confirmed pivots, source semantics:
        # runs every bar, replaces pending while no active position
        if len(zz_vals) >= 3 and active is None:
            v1, v2, v3 = zz_vals[-3], zz_vals[-2], zz_vals[-1]
            d1, d2, d3 = zz_dirs[-3], zz_dirs[-2], zz_dirs[-1]
            is_m = d1 == -1 and d2 == 1 and d3 == -1 and v1 < v2 and v2 > v3 and v1 < v3
            is_w = d1 == 1 and d2 == -1 and d3 == 1 and v1 > v2 and v2 < v3 and v1 > v3
            inv = trade_direction == "inverse"
            if is_m and want_m:
                dist = abs(high[i] - v3)
                a = v3 + dist * tp_pct / 100.0   # above entry
                b = v3 + dist * sl_pct / 100.0   # below entry (sl_pct < 0)
                if exit_mode == "farstop":
                    tp_lv, sl_lv = (b, v1) if inv else (a, v1)
                else:
                    tp_lv, sl_lv = (b, a) if inv else (a, b)
                pending = {"entry": v3, "dir": -1 if inv else 1, "ptype": "M",
                           "v1": v1, "v2": v2, "v3": v3,
                           "tp": tp_lv, "sl": sl_lv}
            elif is_w and want_w:
                dist = abs(v3 - low[i])
                a = v3 - dist * tp_pct / 100.0   # below entry
                b = v3 - dist * sl_pct / 100.0   # above entry
                if exit_mode == "farstop":
                    tp_lv, sl_lv = (b, v1) if inv else (a, v1)
                else:
                    tp_lv, sl_lv = (b, a) if inv else (a, b)
                pending = {"entry": v3, "dir": 1 if inv else -1, "ptype": "W",
                           "v1": v1, "v2": v2, "v3": v3,
                           "tp": tp_lv, "sl": sl_lv}

        # activation: conservative trade-through, not touch
        # default mode: limit orders (long fills below, short fills above)
        # inverse mode: stop orders (long fills above, short fills below)
        entered_this_bar = False
        if active is None and pending is not None:
            d = pending["dir"]
            if trade_direction == "default":
                hit = (d == 1 and low[i] < pending["entry"]) or \
                      (d == -1 and high[i] > pending["entry"])
            else:
                hit = (d == 1 and high[i] > pending["entry"]) or \
                      (d == -1 and low[i] < pending["entry"])
            if hit:
                active = dict(pending, entry_i=i)
                pending = None
                entered_this_bar = True

        # exit management: SL checked FIRST, TP not allowed on the entry bar
        if active is not None:
            exit_px, result = None, 0
            if exit_mode == "time":
                if i - active["entry_i"] >= hold_bars:
                    exit_px = close[i]
                    result = 1 if (exit_px - active["entry"]) * active["dir"] > 0 else -1
            elif active["dir"] == 1:
                if low[i] <= active["sl"]:
                    exit_px, result = active["sl"], -1
                elif not entered_this_bar and high[i] >= active["tp"]:
                    exit_px, result = active["tp"], 1
            else:
                if high[i] >= active["sl"]:
                    exit_px, result = active["sl"], -1
                elif not entered_this_bar and low[i] <= active["tp"]:
                    exit_px, result = active["tp"], 1
            if exit_px is not None:
                pts = (exit_px - active["entry"]) * active["dir"] - 2 * fric_pts
                dollars = pts * POINT_VALUE
                pnl[i] += dollars
                trades.append({
                    "entry_time": idx[active["entry_i"]], "exit_time": idx[i],
                    "dir": active["dir"], "ptype": active["ptype"],
                    "v1": active["v1"], "v2": active["v2"], "v3": active["v3"],
                    "tp_level": active["tp"], "sl_level": active["sl"],
                    "entry": active["entry"], "exit": exit_px,
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


def mw_cpcv(df, param_grid, n_groups=8, k_test=2, embargo_frac=0.01,
            friction_ticks=2.0):
    """CPCV for the event-driven strategy.

    Each param set runs ONCE on the full contiguous series (no seams by
    construction). Per-bar pnl with exit-bar attribution is then summed over
    fold indices: select best params on train indices, score on test indices.
    """
    from .validation import cpcv_paths
    from .sessions import session_group

    pnl_cache = [run_mw(df, friction_ticks=friction_ticks, **p)["pnl"]
                 for p in param_grid]
    paths = cpcv_paths(len(df), n_groups, k_test, embargo_frac)
    rows = []
    for pi, (tr, te) in enumerate(paths):
        best_j = int(np.argmax([pc.iloc[tr].sum() for pc in pnl_cache]))
        te_pnl = pnl_cache[best_j].iloc[te]
        daily = session_group(te_pnl).sum()
        sd = daily.std()
        sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
        rows.append({"path": pi, "params": str(param_grid[best_j]),
                     "oos_pnl": round(float(te_pnl.sum()), 2),
                     "oos_daily_sharpe": round(sharpe, 3)})
    out = pd.DataFrame(rows)
    sh = out["oos_daily_sharpe"].dropna()
    summary = {
        "n_paths": len(out),
        "oos_sharpe_mean": round(float(sh.mean()), 3),
        "oos_sharpe_median": round(float(sh.median()), 3),
        "oos_sharpe_frac_positive": round(float((sh > 0).mean()), 3),
        "oos_pnl_mean": round(float(out["oos_pnl"].mean()), 2),
    }
    return {"paths": out, "summary": summary}
