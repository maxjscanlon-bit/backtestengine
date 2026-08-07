"""Fund-grade validation gate.

Implements:
- Combinatorially Purged Cross-Validation (CPCV) with purging and embargo
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
- Probability of Backtest Overfitting (PBO, CSCV method)

Every candidate strategy must pass this gate before VAL, and results.md must
record n_trials so the Deflated Sharpe reflects everything we actually tried.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps


# ---------------------------------------------------------------- CPCV

def cpcv_paths(n_bars, n_groups=8, k_test=2, embargo_frac=0.01):
    """Generate CPCV train/test index sets.

    Splits the sample into n_groups contiguous groups. Each path holds out
    k_test groups as test, trains on the rest with purge+embargo around
    test boundaries. Returns list of (train_idx, test_idx) arrays.
    """
    edges = np.linspace(0, n_bars, n_groups + 1, dtype=int)
    groups = [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]
    embargo = int(n_bars * embargo_frac)
    paths = []
    for combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in combo])
        mask = np.ones(n_bars, dtype=bool)
        for g in combo:
            lo, hi = edges[g], edges[g + 1]
            plo = max(0, lo - embargo)
            phi = min(n_bars, hi + embargo)
            mask[plo:phi] = False
        train_idx = np.where(mask)[0]
        paths.append((train_idx, np.sort(test_idx)))
    return paths


def cpcv_evaluate(df, signal_fn, param_grid, backtest_fn, n_groups=8, k_test=2,
                  embargo_frac=0.01, friction_ticks=2.0, select_metric="total_pnl"):
    """For each CPCV path: pick best params on train, evaluate on test.

    signal_fn(df, **params) -> position series
    backtest_fn(df, signal, friction_ticks) -> {'stats': {...}, 'pnl': Series}

    Returns dict with per-path out-of-sample stats and the distribution of
    OOS daily Sharpe across paths.
    """
    from .engine import backtest_subset

    paths = cpcv_paths(len(df), n_groups, k_test, embargo_frac)
    # Signals are computed ONCE on the full contiguous series. Slicing the frame
    # before computing rolling features spans seams and fabricates price jumps.
    sig_cache = [signal_fn(df, **p) for p in param_grid]

    oos_rows = []
    for pi, (tr, te) in enumerate(paths):
        best_j, best_score = None, -np.inf
        for j, params in enumerate(param_grid):
            score = backtest_subset(df, sig_cache[j], tr,
                                    friction_ticks=friction_ticks)["stats"].get(select_metric, -np.inf)
            if score > best_score:
                best_score, best_j = score, j
        best_params = param_grid[best_j]
        res = backtest_subset(df, sig_cache[best_j], te, friction_ticks=friction_ticks)
        row = {"path": pi, "params": str(best_params)}
        row.update(res["stats"])
        row["oos_daily_sharpe"] = _daily_sharpe(res["pnl"])
        oos_rows.append(row)
    out = pd.DataFrame(oos_rows)
    sharpes = out["oos_daily_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "n_paths": len(out),
        "oos_sharpe_mean": round(float(sharpes.mean()), 3),
        "oos_sharpe_median": round(float(sharpes.median()), 3),
        "oos_sharpe_frac_positive": round(float((sharpes > 0).mean()), 3),
        "oos_expectancy_mean": round(float(out["expectancy_per_trade"].mean()), 2)
        if "expectancy_per_trade" in out else None,
    }
    return {"paths": out, "summary": summary}


def _daily_sharpe(pnl):
    daily = pnl.groupby(pnl.index.date).sum()
    sd = daily.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(daily.mean() / sd * np.sqrt(252))


# ------------------------------------------------- Deflated Sharpe Ratio

def deflated_sharpe(sr_hat, n_trials, n_obs, skew=0.0, kurt=3.0, sr_benchmark=None):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    sr_hat: observed Sharpe of the selected strategy (per-period, same freq as n_obs)
    n_trials: number of independent strategy trials actually run
    n_obs: number of return observations underlying sr_hat
    skew, kurt: sample skewness and kurtosis (Pearson, normal=3) of returns
    sr_benchmark: if None, computed as the expected max Sharpe of n_trials
                  zero-skill strategies (the correct null under selection)

    Returns dict with dsr (probability the Sharpe is real) and the threshold.
    """
    if sr_benchmark is None:
        # expected maximum of n_trials iid standard normals, scaled by trial variance
        emc = 0.5772156649015329
        if n_trials <= 1:
            max_z = 0.0
        else:
            max_z = ((1 - emc) * sps.norm.ppf(1 - 1.0 / n_trials)
                     + emc * sps.norm.ppf(1 - 1.0 / (n_trials * np.e)))
        sr_benchmark = max_z * np.sqrt(1.0 / max(n_obs - 1, 1))
    denom = np.sqrt(max(1e-12,
                        (1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2) / max(n_obs - 1, 1)))
    z = (sr_hat - sr_benchmark) / denom
    return {
        "dsr": round(float(sps.norm.cdf(z)), 4),
        "sr_hat": round(float(sr_hat), 4),
        "sr_benchmark": round(float(sr_benchmark), 4),
        "n_trials": n_trials,
        "n_obs": n_obs,
    }


def sharpe_moments(returns):
    """Per-period Sharpe plus skew and kurtosis, for feeding deflated_sharpe."""
    r = pd.Series(returns).dropna()
    sd = r.std()
    sr = float(r.mean() / sd) if sd > 0 else np.nan
    return sr, float(sps.skew(r)), float(sps.kurtosis(r, fisher=False)), len(r)


# ----------------------------------------------------------------- PBO

def pbo_cscv(perf_matrix, n_splits=16):
    """Probability of Backtest Overfitting via CSCV.

    perf_matrix: DataFrame, rows = time chunks (equal length), cols = strategy trials,
                 values = pnl per chunk per trial.
    Splits chunks into two halves in all C(n_splits, n/2) combinations, picks the
    in-sample best trial, measures its out-of-sample rank. PBO = fraction of
    combinations where the IS best is below median OOS.
    """
    m = perf_matrix.values
    n_chunks, n_trials = m.shape
    if n_chunks < n_splits:
        n_splits = n_chunks - (n_chunks % 2)
    # aggregate chunks into n_splits blocks
    edges = np.linspace(0, n_chunks, n_splits + 1, dtype=int)
    blocks = np.array([m[edges[i]:edges[i + 1]].sum(axis=0) for i in range(n_splits)])
    half = n_splits // 2
    logits = []
    for combo in combinations(range(n_splits), half):
        is_idx = np.array(combo)
        oos_idx = np.array([i for i in range(n_splits) if i not in combo])
        is_perf = blocks[is_idx].sum(axis=0)
        oos_perf = blocks[oos_idx].sum(axis=0)
        best = int(np.argmax(is_perf))
        rank = sps.rankdata(oos_perf)[best] / (n_trials + 1)
        rank = min(max(rank, 1e-9), 1 - 1e-9)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {
        "pbo": round(float((logits < 0).mean()), 4),
        "n_combinations": len(logits),
        "median_oos_rank_of_is_best": round(float(sps.norm.cdf(np.median(logits))), 4),
    }


def build_perf_matrix(df, signal_fn, param_grid, backtest_fn, n_chunks=16,
                      friction_ticks=2.0):
    """Run every trial over the full period, return chunked pnl matrix for PBO."""
    n = len(df)
    edges = np.linspace(0, n, n_chunks + 1, dtype=int)
    cols = {}
    for j, params in enumerate(param_grid):
        sig = signal_fn(df, **params)
        pnl = backtest_fn(df, sig, friction_ticks=friction_ticks)["pnl"].values
        cols[f"trial_{j}"] = [pnl[edges[i]:edges[i + 1]].sum() for i in range(n_chunks)]
    return pd.DataFrame(cols)
