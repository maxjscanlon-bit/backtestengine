"""Statistical validation: permutation tests and sub-period stability."""

import numpy as np
import pandas as pd
from .engine import backtest


def permutation_test(df, signal, n_perm=1000, friction_ticks=2.0, seed=42):
    """Shuffle signal values (block-free baseline) and build the null distribution
    of total net pnl. Real result must sit in the right tail.

    Returns dict with real pnl, null mean/std, and one-sided p-value.
    """
    rng = np.random.default_rng(seed)
    real = backtest(df, signal, friction_ticks=friction_ticks)["stats"]["total_pnl"]
    sig = signal.reindex(df.index).fillna(0.0)
    vals = sig.values.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(vals)
        shuffled = pd.Series(vals.copy(), index=sig.index)
        null[i] = backtest(df, shuffled, friction_ticks=friction_ticks)["stats"]["total_pnl"]
    p = float((null >= real).mean())
    return {
        "real_pnl": round(real, 2),
        "null_mean": round(float(null.mean()), 2),
        "null_std": round(float(null.std()), 2),
        "p_value_one_sided": round(p, 4),
        "n_perm": n_perm,
    }


def subperiod_stability(df, signal, n_chunks=4, friction_ticks=2.0):
    """Split the period into chunks and check the edge is not one regime's artifact."""
    n = len(df)
    rows = []
    for k in range(n_chunks):
        lo = int(n * k / n_chunks)
        hi = int(n * (k + 1) / n_chunks)
        chunk = df.iloc[lo:hi]
        r = backtest(chunk, signal.reindex(chunk.index).fillna(0.0),
                     friction_ticks=friction_ticks)
        row = {"chunk": k + 1,
               "start": str(chunk.index.min().date()),
               "end": str(chunk.index.max().date())}
        row.update(r["stats"])
        rows.append(row)
    return pd.DataFrame(rows)
