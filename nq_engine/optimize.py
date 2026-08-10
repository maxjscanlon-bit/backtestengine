"""Parameter optimizer, fund-style.

Given a signal function and a parameter space, this does NOT return "the best
parameters." It returns:
  1. CPCV out-of-sample performance of the selection process itself
  2. A ranked table of parameter sets by full-TRAIN performance, each annotated
     with neighborhood plateau stats (median performance of adjacent settings)
  3. Total trial count for Deflated Sharpe accounting
  4. Plateau-vs-spike flags

Usage:
    from nq_engine.optimize import optimize
    report = optimize(train_df, signal_fn, space, backtest)
"""

from itertools import product

import numpy as np
import pandas as pd

from .sessions import session_group
from .validation import cpcv_evaluate, deflated_sharpe, sharpe_moments


def expand_grid(space):
    """space: dict of param -> list of values. Returns list of param dicts."""
    keys = list(space.keys())
    return [dict(zip(keys, combo)) for combo in product(*[space[k] for k in keys])]


def _neighbors(params, space):
    """Parameter sets one step away in any single numeric dimension."""
    out = []
    for k, vals in space.items():
        if params[k] not in vals:
            continue
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(params)
                q[k] = vals[j]
                out.append(q)
    return out


def optimize(df, signal_fn, space, backtest_fn, friction_ticks=2.0,
             n_groups=8, k_test=2, metric="expectancy_per_trade",
             prior_trials=0):
    """Run the full honest optimization. df should be TRAIN only."""
    grid = expand_grid(space)
    n_trials = len(grid) + prior_trials

    # ---- full-TRAIN scan (for ranking and plateau analysis only)
    rows = []
    cache = {}
    for params in grid:
        key = tuple(sorted(params.items()))
        r = backtest_fn(df, signal_fn(df, **params), friction_ticks=friction_ticks)
        cache[key] = r
        row = dict(params)
        row.update(r["stats"])
        rows.append(row)
    scan = pd.DataFrame(rows)

    # ---- plateau annotation
    plateau_med, plateau_flag = [], []
    for params in grid:
        nb = _neighbors(params, space)
        vals = []
        for q in nb:
            key = tuple(sorted(q.items()))
            if key in cache:
                vals.append(cache[key]["stats"].get(metric, np.nan))
        med = float(np.nanmedian(vals)) if vals else np.nan
        plateau_med.append(round(med, 2) if med == med else np.nan)
        own = cache[tuple(sorted(params.items()))]["stats"].get(metric, np.nan)
        # spike flag: own value positive but neighborhood median negative or
        # less than a third of own
        spike = bool(own > 0 and (med != med or med <= 0 or med < own / 3))
        plateau_flag.append("SPIKE" if spike else ("plateau" if own > 0 else ""))
    scan["neighbor_median"] = plateau_med
    scan["shape"] = plateau_flag
    scan = scan.sort_values(metric, ascending=False).reset_index(drop=True)

    # ---- CPCV: honest out-of-sample performance of selecting from this grid
    cp = cpcv_evaluate(df, signal_fn, grid, backtest_fn, n_groups=n_groups,
                       k_test=k_test, friction_ticks=friction_ticks,
                       select_metric="total_pnl")

    # ---- Deflated Sharpe of the top full-TRAIN pick, penalized by n_trials
    top = scan.iloc[0]
    top_params = {k: top[k] for k in space}
    r_top = cache[tuple(sorted(top_params.items()))]
    daily = session_group(r_top["pnl"]).sum()
    sr, sk, ku, n = sharpe_moments(daily)
    dsr = deflated_sharpe(sr, n_trials=n_trials, n_obs=n, skew=sk, kurt=ku)

    return {
        "scan": scan,
        "cpcv": cp,
        "dsr_top": dsr,
        "n_trials": n_trials,
        "top_params": top_params,
        "verdict": _verdict(cp["summary"], dsr, scan),
    }


def _verdict(cpcv_summary, dsr, scan):
    """Mechanical verdict. Human judgment still applies on top."""
    oos_ok = (cpcv_summary["oos_sharpe_frac_positive"] >= 0.7
              and cpcv_summary["oos_sharpe_mean"] > 0.5)
    dsr_ok = dsr["dsr"] >= 0.95
    top_is_plateau = scan.iloc[0]["shape"] == "plateau"
    if oos_ok and dsr_ok and top_is_plateau:
        return "CANDIDATE: passes gate on TRAIN. Eligible for VAL."
    reasons = []
    if not oos_ok:
        reasons.append("CPCV OOS weak")
    if not dsr_ok:
        reasons.append(f"DSR {dsr['dsr']} below 0.95")
    if not top_is_plateau:
        reasons.append("top pick is a spike, not a plateau")
    return "REJECT: " + "; ".join(reasons)
