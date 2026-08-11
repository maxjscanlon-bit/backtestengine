"""Independent verification of the Cypher engine using backtesting.py.

This is a THIRD implementation, written against a library I did not write, with
its own order lifecycle, fill logic, and accounting. Agreement between it and
nq_engine is evidence that is not reachable by testing my code against my code.

Known convention differences, documented rather than papered over:
  - backtesting.py fills limit orders when price TOUCHES the level. Our engine
    requires trade-through. We therefore expect a small number of extra fills.
  - backtesting.py resolves same-bar SL/TP by checking SL first for longs, which
    matches our pessimistic assumption.
  - backtesting.py has no concept of "no TP on the entry bar". Our engine
    forbids it. This is the main expected divergence.
  - backtesting.py works in whole units and applies commission as a fraction of
    notional, so friction is set to 0 here and compared on GROSS points.
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


def cypher_levels(df, period=10, atr_n=96, atr_sl=1.5, atr_tp=1.0,
                  b_lo=0.382, b_hi=0.618, c_lo=1.272, c_hi=1.414, d_ret=0.786):
    """Precompute pattern signals as arrays: entry, sl, tp, dir per bar.

    Deliberately computed by walking the series, same as the reference engine,
    but kept separate so backtesting.py owns all order handling.
    """
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    pc = pd.Series(close).shift(1)
    tr = pd.concat([pd.Series(high) - pd.Series(low),
                    (pd.Series(high) - pc).abs(),
                    (pd.Series(low) - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_n, adjust=False).mean().shift(1).values

    hh = pd.Series(high).rolling(period - 1).max().shift(1).values
    ll = pd.Series(low).rolling(period - 1).min().shift(1).values

    sig_entry = np.full(n, np.nan)
    sig_sl = np.full(n, np.nan)
    sig_tp = np.full(n, np.nan)
    sig_dir = np.zeros(n)

    zz_v, zz_d = [], []
    cur_val, cur_dir, dir_ = None, 0, 0
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
        if len(zz_v) >= 4:
            X, A, B, C = zz_v[-4], zz_v[-3], zz_v[-2], zz_v[-1]
            dX, dA, dB, dC = zz_d[-4], zz_d[-3], zz_d[-2], zz_d[-1]
            xa = A - X
            bull = dX == -1 and dA == 1 and dB == -1 and dC == 1 and xa > 0
            bear = dX == 1 and dA == -1 and dB == 1 and dC == -1 and xa < 0
            if bull or bear:
                rb = (A - B) / xa
                rc = (C - X) / xa
                if b_lo <= rb <= b_hi and c_lo <= rc <= c_hi:
                    D = C - d_ret * (C - X)
                    d = 1 if bull else -1
                    sig_entry[i] = D
                    sig_sl[i] = D - d * atr_sl * atr[i]
                    sig_tp[i] = D + d * atr_tp * atr[i]
                    sig_dir[i] = d
    return sig_entry, sig_sl, sig_tp, sig_dir


class CypherBT(Strategy):
    entry_arr = None
    sl_arr = None
    tp_arr = None
    dir_arr = None

    def init(self):
        pass

    def next(self):
        i = len(self.data) - 1
        d = self.dir_arr[i]
        if d == 0 or self.position:
            return
        entry = self.entry_arr[i]
        sl = self.sl_arr[i]
        tp = self.tp_arr[i]
        if np.isnan(entry):
            return
        # cancel any resting orders, one live setup at a time
        for o in list(self.orders):
            o.cancel()
        price = self.data.Close[-1]
        try:
            if d == 1 and entry < price:
                self.buy(limit=entry, sl=sl, tp=tp, size=1)
            elif d == -1 and entry > price:
                self.sell(limit=entry, sl=sl, tp=tp, size=1)
        except Exception:
            pass


def run_bt(df5):
    """df5: our OHLCV frame. Returns backtesting.py trade list in points."""
    d = df5.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"}).copy()
    e, s, t, dr = cypher_levels(d)
    CypherBT.entry_arr, CypherBT.sl_arr, CypherBT.tp_arr, CypherBT.dir_arr = e, s, t, dr
    bt = Backtest(d, CypherBT, cash=10_000_000, commission=0.0,
                  trade_on_close=False, exclusive_orders=True)
    stats = bt.run()
    return stats
