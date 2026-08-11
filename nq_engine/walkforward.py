"""Walk-forward parameter selection."""
import numpy as np, pandas as pd

def walk_forward(df, runner, param_grid, n_windows=8, train_frac=0.7,
                 metric="total_pnl", **run_kw):
    """Rolling anchored walk-forward.

    Splits the sample into n_windows sequential blocks. For each, fits params on
    the first train_frac of the block and applies them to the remainder. Reports
    OOS results only. This measures the SELECTION PROCESS, not a fixed strategy.
    """
    n = len(df)
    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    rows = []
    for k in range(n_windows):
        lo, hi = edges[k], edges[k + 1]
        cut = lo + int((hi - lo) * train_frac)
        tr, te = df.iloc[lo:cut], df.iloc[cut:hi]
        if len(tr) < 5000 or len(te) < 2000:
            continue
        best, best_s = None, -np.inf
        for p in param_grid:
            s = runner(tr, **{**run_kw, **p})["stats"].get(metric, -np.inf)
            if s > best_s:
                best_s, best = s, p
        r = runner(te, **{**run_kw, **best})
        st = r["stats"]
        rows.append({"window": k + 1,
                     "oos_start": str(te.index.min().date()),
                     "oos_end": str(te.index.max().date()),
                     "params": str(best),
                     "is_pnl": round(best_s, 0),
                     "oos_pnl": round(st.get("total_pnl", 0), 0),
                     "oos_trades": st.get("n_trades", 0),
                     "oos_avg": st.get("expectancy_per_trade", np.nan)})
    out = pd.DataFrame(rows)
    summary = {
        "windows": len(out),
        "oos_total": round(float(out["oos_pnl"].sum()), 0),
        "oos_windows_positive": round(float((out["oos_pnl"] > 0).mean()), 3),
        "param_stability": out["params"].nunique(),
    }
    return {"windows": out, "summary": summary}
