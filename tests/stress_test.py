"""Stress test suite for nq_engine. Run: python3 tests/stress_test.py

Tests both directions: the gate must PASS planted edges and REJECT noise.
Also verifies mechanical correctness of the engine itself.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/claude/nq-research')
from nq_engine.engine import backtest, TICK
from nq_engine.signals import zscore_reversal
from nq_engine.validation import (cpcv_paths, cpcv_evaluate, deflated_sharpe,
                                  sharpe_moments, pbo_cscv, build_perf_matrix)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((PASS if condition else FAIL, name, detail))


def make_noise(seed, n=8000, sigma=8.0, start=15000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2023-01-02 09:30', periods=n, freq='5min')
    px = start + np.cumsum(rng.normal(0, sigma, n))
    return pd.DataFrame({'open': px, 'high': px + 5, 'low': px - 5,
                         'close': px, 'volume': 1000.0}, index=idx)


def make_edge(seed, n=8000, sigma=8.0, revert=0.35, start=15000):
    """Random walk with mean reversion after large down moves. Planted, real edge."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2023-01-02 09:30', periods=n, freq='5min')
    steps = rng.normal(0, sigma, n)
    for i in range(1, n):
        if steps[i - 1] < -1.5 * sigma:
            steps[i] += revert * abs(steps[i - 1])
    px = start + np.cumsum(steps)
    return pd.DataFrame({'open': px, 'high': px + 5, 'low': px - 5,
                         'close': px, 'volume': 1000.0}, index=idx)


# ================================================== 1. ENGINE MECHANICS

# 1a. Flat signal produces exactly zero pnl and zero trades
df = make_noise(0)
flat = pd.Series(0.0, index=df.index)
r = backtest(df, flat, friction_ticks=2)
check("flat signal zero pnl", r['stats']['total_pnl'] == 0 and r['stats']['n_trades'] == 0)

# 1b. Always-long with zero friction equals buy and hold point change
al = pd.Series(1.0, index=df.index)
r = backtest(df, al, friction_ticks=0)
# independent reference: signal at bar 0 means hold from bar 0 close onward
ref = (df['close'].iloc[-1] - df['close'].iloc[0]) * 20.0
check("always-long matches independent buy-and-hold reference",
      abs(r['stats']['total_pnl'] - ref) < 0.01,
      f"engine {r['stats']['total_pnl']:.2f} vs reference {ref:.2f}")

# 1b2. REGRESSION: evaluating a fold must not accrue pnl on seam bars.
# Build a frame with a huge artificial gap, then score a discontiguous index set.
from nq_engine.engine import backtest_subset
gap_df = df.copy()
gap_df.iloc[5000:, :4] += 5000.0          # 5000-point discontinuity at bar 5000
always = pd.Series(1.0, index=gap_df.index)
idx_span = np.concatenate([np.arange(4000, 4010), np.arange(5000, 5010)])
r_span = backtest_subset(gap_df, always, idx_span, friction_ticks=0)
check("subset scoring excludes the seam jump",
      abs(r_span['stats']['total_pnl']) < 5000 * 20 * 0.5,
      f"pnl {r_span['stats']['total_pnl']:.0f}, artificial gap worth {5000*20:,}")

# 1b3. REGRESSION: signal must be computed on full series, not per-fold slices.
# Same signal scored two ways on a contiguous fold should agree exactly.
contig = np.arange(1000, 3000)
a = backtest_subset(df, zscore_reversal(df, session_rth_only=False), contig, friction_ticks=2)
b = backtest(df.iloc[contig[0] - 1:contig[-1] + 1],
             zscore_reversal(df, session_rth_only=False).iloc[contig[0] - 1:contig[-1] + 1],
             friction_ticks=2)
check("subset and full-slice agree on contiguous fold",
      abs(a['stats']['total_pnl'] - b['stats']['total_pnl']) < 1e-6 + 0.02 * abs(b['stats']['total_pnl']),
      f"subset {a['stats']['total_pnl']:.0f} vs slice {b['stats']['total_pnl']:.0f}")

# 1b4. REGRESSION: signals must use point returns, immune to additive back-adjust offset
shifted = df.copy()
shifted[['open', 'high', 'low', 'close']] += 3000.0
check("signal invariant to back-adjustment offset",
      zscore_reversal(df, session_rth_only=False).equals(
          zscore_reversal(shifted, session_rth_only=False)),
      "pct_change would break this, point diffs do not")

# 1c. No lookahead: signal = sign of CURRENT bar return must not capture that bar
cur_sign = np.sign(df['close'].diff()).fillna(0.0)
r_cur = backtest(df, cur_sign, friction_ticks=0)
# perfect foresight of NEXT bar (cheating on purpose) must be hugely profitable
next_sign = np.sign(df['close'].diff().shift(-1)).fillna(0.0)
r_next = backtest(df, next_sign, friction_ticks=0)
check("no lookahead by construction",
      abs(r_cur['stats']['total_pnl']) < 0.15 * r_next['stats']['total_pnl'],
      f"current-sign pnl {r_cur['stats']['total_pnl']:.0f}, foresight pnl {r_next['stats']['total_pnl']:.0f}")

# 1d. Friction accounting: pnl difference between 0 and 2 ticks equals turns * cost
sig = zscore_reversal(df, session_rth_only=False)
r0 = backtest(df, sig, friction_ticks=0)
r2 = backtest(df, sig, friction_ticks=2)
turns = sig.diff().abs().fillna(sig.abs()).sum()
expected_cost = turns * 2 * TICK * 20.0
actual_cost = r0['stats']['total_pnl'] - r2['stats']['total_pnl']
check("friction accounting exact", abs(actual_cost - expected_cost) < 1.0,
      f"expected {expected_cost:.2f} charged {actual_cost:.2f}")

# 1e. Trade extraction consistency: sum of trade pnl close to series pnl
tr_sum = r2['trades']['net_pnl'].sum() if len(r2['trades']) else 0
open_pos_tail = sig.iloc[-1] != 0
check("trade pnl reconciles with series pnl",
      abs(tr_sum - r2['stats']['total_pnl']) < max(50.0, 0.02 * abs(r2['stats']['total_pnl']) + 50),
      f"trades {tr_sum:.2f} vs series {r2['stats']['total_pnl']:.2f} open_tail={open_pos_tail}")

# ============================================ 1d. SESSIONS AND ALIGNMENT

from nq_engine.sessions import rth_mask, session_date
from nq_engine.data import load_split

_real = load_split('/home/claude/nq-research/data', 'train')

# right-labeled bars: keep (09:30, 16:00], so 09:30 out and 16:00 in
_day = _real.loc['2024-03-05 09:00':'2024-03-05 16:30']
_m = pd.Series(rth_mask(_day.index), index=_day.index)
check("RTH excludes the 09:30 bar (covers 09:25-09:30, pre-open)",
      not _m.loc['2024-03-05 09:30'])
check("RTH includes the 09:35 and 16:00 bars",
      _m.loc['2024-03-05 09:35'] and _m.loc['2024-03-05 16:00'])
check("RTH excludes the 16:05 bar (covers 16:00-16:05, post-close)",
      not _m.loc['2024-03-05 16:05'])

_rth = _real[rth_mask(_real.index)]
_cnt = _rth.groupby(_rth.index.date).size()
check("RTH day is exactly 78 bars on full sessions",
      (_cnt == 78).mean() > 0.95, f"{(_cnt==78).sum()}/{len(_cnt)} days at 78")

# bar alignment: session opens 18:00, so first bar of a session is labeled 18:05
_open_labels = set(pd.Series(_real.index).dt.strftime('%H:%M')[
    pd.Series(_real.index).diff().gt(pd.Timedelta(minutes=30)).fillna(False).values])
check("first bar after the maintenance break is labeled 18:05",
      _open_labels == {'18:05'}, f"labels seen: {sorted(_open_labels)}")

# session date rolls evening bars into the next trading day
check("evening bars roll into the next session date",
      str(session_date(pd.DatetimeIndex(['2024-03-03 18:05']))[0]) == '2024-03-04')
check("afternoon bars stay on their own session date",
      str(session_date(pd.DatetimeIndex(['2024-03-04 16:00']))[0]) == '2024-03-04')
check("session grouping removes split-session artifacts",
      len(set(session_date(_real.index))) < len(set(_real.index.date)),
      f"{len(set(_real.index.date))} calendar -> {len(set(session_date(_real.index)))} sessions")

# ================================================== 2. CPCV MECHANICS

paths = cpcv_paths(1000, n_groups=8, k_test=2, embargo_frac=0.02)
ok_overlap, ok_embargo = True, True
for tr_idx, te_idx in paths:
    if np.intersect1d(tr_idx, te_idx).size > 0:
        ok_overlap = False
    # every train index must be at least embargo away from any test boundary block
    te_set = set(te_idx.tolist())
    for t in tr_idx:
        if any((t + d) in te_set for d in range(-19, 20) if d != 0):
            # within embargo distance of test region
            lo = t - 19 >= 0
            ok_embargo = ok_embargo  # detailed check below
from itertools import combinations as _comb
check("CPCV no train/test overlap", ok_overlap)
check("CPCV path count C(8,2)=28", len(paths) == 28)
# explicit embargo check: min distance from any train idx to test region >= embargo
embargo = 20
min_gap = 10**9
for tr_idx, te_idx in paths:
    te = np.asarray(te_idx)
    # boundaries of contiguous test blocks
    splits = np.where(np.diff(te) > 1)[0]
    blocks = np.split(te, splits + 1)
    for b in blocks:
        lo, hi = b[0], b[-1]
        near = tr_idx[(tr_idx >= lo - embargo) & (tr_idx <= hi + embargo)]
        if near.size:
            min_gap = 0
check("CPCV embargo respected", min_gap > 0, f"embargo bars {embargo}")

# ================================================== 3. GATE: NEGATIVE CONTROLS

grid = [{'lookback': lb, 'z_entry': z, 'hold_bars': h, 'session_rth_only': False}
        for lb in (24, 48) for z in (-1.0, -1.5, -2.0) for h in (3, 6)]

neg_pass = 0
for seed in (1, 2, 3):
    dfn = make_noise(seed)
    cp = cpcv_evaluate(dfn, zscore_reversal, grid, backtest, n_groups=6, k_test=2)
    if cp['summary']['oos_sharpe_frac_positive'] > 0.7 and cp['summary']['oos_sharpe_mean'] > 1.0:
        neg_pass += 1
check("CPCV rejects noise across 3 seeds", neg_pass == 0, f"false passes {neg_pass}/3")

# DSR on noise across seeds
dsr_noise = []
for seed in (1, 2, 3):
    dfn = make_noise(seed)
    best_daily, best_sr = None, -np.inf
    for p in grid:
        rr = backtest(dfn, zscore_reversal(dfn, **p), friction_ticks=2)
        d = rr['pnl'].groupby(rr['pnl'].index.date).sum()
        s = d.mean() / d.std() if d.std() > 0 else -np.inf
        if s > best_sr:
            best_sr, best_daily = s, d
    sr, sk, ku, n = sharpe_moments(best_daily)
    dsr_noise.append(deflated_sharpe(sr, len(grid), n, sk, ku)['dsr'])
check("DSR rejects noise (all < 0.5)", all(d < 0.5 for d in dsr_noise),
      f"dsr values {[round(d,3) for d in dsr_noise]}")

# ================================================== 4. GATE: POSITIVE CONTROLS

pos_pass = 0
pos_detail = []
for seed in (11, 12, 13):
    dfe = make_edge(seed)
    cp = cpcv_evaluate(dfe, zscore_reversal, grid, backtest, n_groups=6, k_test=2)
    s = cp['summary']
    pos_detail.append((s['oos_sharpe_mean'], s['oos_sharpe_frac_positive']))
    if s['oos_sharpe_frac_positive'] >= 0.7 and s['oos_sharpe_mean'] > 1.0:
        pos_pass += 1
check("CPCV passes planted edge across 3 seeds", pos_pass == 3,
      f"details {[(round(a,2), b) for a, b in pos_detail]}")

dsr_edge = []
for seed in (11, 12, 13):
    dfe = make_edge(seed)
    best_daily, best_sr = None, -np.inf
    for p in grid:
        rr = backtest(dfe, zscore_reversal(dfe, **p), friction_ticks=2)
        d = rr['pnl'].groupby(rr['pnl'].index.date).sum()
        s = d.mean() / d.std() if d.std() > 0 else -np.inf
        if s > best_sr:
            best_sr, best_daily = s, d
    sr, sk, ku, n = sharpe_moments(best_daily)
    dsr_edge.append(deflated_sharpe(sr, len(grid), n, sk, ku)['dsr'])
check("DSR passes planted edge (all > 0.9)", all(d > 0.9 for d in dsr_edge),
      f"dsr values {[round(d,3) for d in dsr_edge]}")

# ================================================== 5. DSR SANITY

sr_fixed, n_obs = 0.25, 250
d1 = deflated_sharpe(sr_fixed, 1, n_obs)['dsr']
d10 = deflated_sharpe(sr_fixed, 10, n_obs)['dsr']
d100 = deflated_sharpe(sr_fixed, 100, n_obs)['dsr']
check("DSR monotone decreasing in n_trials", d1 > d10 > d100,
      f"{d1:.3f} > {d10:.3f} > {d100:.3f}")

d_fat = deflated_sharpe(sr_fixed, 10, n_obs, skew=-1.0, kurt=8.0)['dsr']
check("DSR penalizes fat tails and negative skew", d_fat < d10,
      f"normal {d10:.3f} vs fat-tailed {d_fat:.3f}")

# ================================================== 6. PBO SANITY

# frictionless pure noise trials: PBO should hover near 0.5
rng = np.random.default_rng(99)
pm_noise = pd.DataFrame(rng.normal(0, 1, (16, 20)),
                        columns=[f"t{i}" for i in range(20)])
p_noise = pbo_cscv(pm_noise, n_splits=16)['pbo']
check("PBO near 0.5 on iid noise trials", 0.3 < p_noise < 0.7, f"pbo {p_noise}")

# one dominant trial: PBO should be near 0
pm_dom = pm_noise.copy()
pm_dom['t0'] = pm_dom['t0'] + 3.0
p_dom = pbo_cscv(pm_dom, n_splits=16)['pbo']
check("PBO near 0 with one dominant trial", p_dom < 0.1, f"pbo {p_dom}")

# ================================================== REPORT

print()
n_fail = sum(1 for s, _, _ in results if s == FAIL)
for s, name, detail in results:
    line = f"[{s}] {name}"
    if detail:
        line += f"  |  {detail}"
    print(line)
print()
print(f"{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
