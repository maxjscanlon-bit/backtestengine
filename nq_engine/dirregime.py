"""Directional (bull/bear) regime classifiers, institutional standard set.

All classifiers:
  - operate on a DAILY series built from session closes
  - are point-based and offset-invariant (Panama-safe): comparisons of price to
    moving averages survive additive shifts; anything ratio-like is expressed
    in daily-ATR units instead of percent
  - are shifted so the state for day t uses information through day t-1 only

State encoding: +1 bull, -1 bear, 0 warmup/undefined.
"""

import numpy as np
import pandas as pd


def daily_from_sessions(df5, rth_only=True):
    """Collapse 5-min bars into a daily session frame (open, high, low, close)."""
    import datetime as dt
    from .sessions import session_date
    d = df5.copy()
    d["sess"] = session_date(d.index)
    if rth_only:
        tod = d.index.time
        d = d[(tod > dt.time(9, 30)) & (tod <= dt.time(16, 0))]
    rows = []
    for sess, g in d.groupby("sess"):
        if len(g) < 30:
            continue
        rows.append(dict(sess=pd.Timestamp(sess), open=g["open"].iloc[0],
                         high=g["high"].max(), low=g["low"].min(),
                         close=g["close"].iloc[-1]))
    return pd.DataFrame(rows).set_index("sess").sort_index()


def _datr(D, n=20):
    pc = D["close"].shift(1)
    tr = pd.concat([D["high"] - D["low"], (D["high"] - pc).abs(),
                    (D["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def classifiers(D):
    """Return DataFrame of {-1, 0, +1} states, one column per classifier.
    Every column is shifted one day: state[t] is known at t-1 close."""
    c = D["close"]
    out = {}

    out["sma200"] = np.sign(c - c.rolling(200).mean())
    out["sma50"] = np.sign(c - c.rolling(50).mean())
    out["golden_cross"] = np.sign(c.rolling(50).mean() - c.rolling(200).mean())
    out["tsmom_12m"] = np.sign(c - c.shift(250))
    out["tsmom_3m"] = np.sign(c - c.shift(63))
    out["tsmom_1m"] = np.sign(c - c.shift(21))
    out["ma100_slope"] = np.sign(c.rolling(100).mean().diff(10))
    out["ewma_20_100"] = np.sign(c.ewm(span=20, adjust=False).mean()
                                 - c.ewm(span=100, adjust=False).mean())
    mid = (c.rolling(126).max() + c.rolling(126).min()) / 2
    out["donchian_mid"] = np.sign(c - mid)

    # drawdown with hysteresis, in daily-ATR units (offset invariant):
    # bear once >= 4 daily ATR below the 250d high, bull once within 1 ATR
    atr = _datr(D)
    dd = (c.rolling(250).max() - c) / atr
    st = np.zeros(len(c))
    cur = 0
    for i in range(len(c)):
        v = dd.iloc[i]
        if np.isnan(v):
            st[i] = 0
            continue
        if v >= 4.0:
            cur = -1
        elif v <= 1.0:
            cur = 1
        st[i] = cur
    out["dd_hysteresis"] = pd.Series(st, index=c.index)

    X = pd.DataFrame(out, index=c.index)
    return X.shift(1).fillna(0).astype(int)


def hmm_state(D, refit_every=21, min_hist=300, seed=0):
    """2-state Gaussian HMM on daily point returns, expanding window, refit
    monthly. State for day t is decoded using a model fit on data through the
    most recent refit date strictly before t. Bull = the state with the higher
    fitted mean. Returns a shifted series like the other classifiers."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None
    r = D["close"].diff().values
    n = len(r)
    state = np.zeros(n)
    model = None
    bull_ix = 1
    for t in range(min_hist, n):
        if (t - min_hist) % refit_every == 0:
            hist = r[1:t].reshape(-1, 1)
            model = GaussianHMM(n_components=2, covariance_type="diag",
                                n_iter=100, random_state=seed)
            try:
                model.fit(hist)
                bull_ix = int(np.argmax(model.means_.ravel()))
            except Exception:
                model = None
        if model is not None:
            try:
                z = model.predict(r[1:t + 1].reshape(-1, 1))[-1]
                state[t] = 1 if z == bull_ix else -1
            except Exception:
                state[t] = 0
    return pd.Series(state, index=D.index).shift(1).fillna(0).astype(int)
