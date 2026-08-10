"""Markov meta-layer for the M/W strategy.

Design: the unfiltered strategy runs as a SHADOW ledger (every pattern trade
simulated, exactly what the NT pseudo-engine does). The meta-layer takes a
real position only when the recent shadow outcome state qualifies. Because the
event engine is strictly sequential (next entry bar > previous exit bar),
every decision conditions only on closed shadow trades. No lookahead.

Rules:
  ('afterW', 1)        take only if the last shadow trade won
  ('afterW', 2)        take only if the last two shadow trades both won
  ('roll', (k, thr))   take only if rolling shadow win-rate over last k >= thr
"""

import numpy as np
import pandas as pd


def apply_meta(trades, index, rule):
    """Filter a shadow trade list by a Markov state rule.

    trades: DataFrame from run_mw (shadow ledger), must be time-ordered
    index: bar index of the full df, for rebuilding the pnl series
    rule: tuple as documented above

    Returns {'trades', 'pnl', 'taken_mask'}.
    """
    t = trades.sort_values('exit_time').reset_index(drop=True)
    w = (t['result'] == 1).astype(int).values
    n = len(t)
    take = np.zeros(n, dtype=bool)

    kind = rule[0]
    if kind == 'afterW':
        m = rule[1]
        for i in range(n):
            if i >= m and w[i - m:i].all():
                take[i] = True
    elif kind == 'roll':
        k, thr = rule[1]
        for i in range(n):
            if i >= k and w[i - k:i].mean() >= thr:
                take[i] = True
    else:
        raise ValueError(rule)

    taken = t[take].copy()
    pnl = pd.Series(0.0, index=index)
    if len(taken):
        add = taken.groupby('exit_time')['net_pnl'].sum()
        pnl.loc[add.index] += add.values
    return {'trades': taken, 'pnl': pnl, 'taken_mask': take}


def meta_stats(res, index):
    from .engine import summarize
    equity = res['pnl'].cumsum()
    return summarize(res['pnl'], equity, res['trades'])
