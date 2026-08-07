"""Build a continuous, back-adjusted futures series from Databento parent-symbol
OHLCV data (all contract months in one file).

Steps:
1. Filter to outright quarterly contracts (drop calendar spreads)
2. Determine the front contract each day by rolling dominant volume
   (roll when a later-expiry contract's daily volume exceeds the current front's)
3. Back-adjust: at each roll, shift all earlier history by the close-to-close
   price difference between old and new contract at the roll boundary
4. Convert UTC to US/Eastern, aggregate 1-min to N-min bars
"""

import re
import numpy as np
import pandas as pd

OUTRIGHT_RE = re.compile(r"^NQ[HMUZ]\d$")

MONTH_CODE = {"H": 3, "M": 6, "U": 9, "Z": 12}


def _expiry_rank(sym):
    """Sortable expiry key from symbol like NQU1 (year digit ambiguity resolved
    by decade window 2020s)."""
    m, y = sym[2], int(sym[3])
    year = 2020 + y if y >= 1 else 2030
    return year * 100 + MONTH_CODE[m]


def load_databento(path):
    df = pd.read_csv(path, usecols=["ts_event", "open", "high", "low", "close",
                                    "volume", "symbol"])
    df = df[df["symbol"].str.match(OUTRIGHT_RE)]
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event").sort_index()
    return df


def build_continuous(df, roll_confirm_days=2):
    """Volume-based roll with back-adjustment. Returns continuous 1-min frame."""
    daily_vol = (df.groupby([pd.Grouper(freq="1D"), "symbol"])["volume"]
                   .sum().unstack(fill_value=0))
    front_by_day = {}
    current = None
    beat_days = 0
    challenger = None
    for day, row in daily_vol.iterrows():
        active = row[row > 0]
        if active.empty:
            continue
        if current is None or current not in active.index:
            current = active.idxmax()
            beat_days, challenger = 0, None
        else:
            top = active.idxmax()
            if top != current and _expiry_rank(top) > _expiry_rank(current):
                if top == challenger:
                    beat_days += 1
                else:
                    challenger, beat_days = top, 1
                if beat_days >= roll_confirm_days:
                    current = top
                    beat_days, challenger = 0, None
            else:
                beat_days, challenger = 0, None
        front_by_day[day.normalize()] = current

    fmap = pd.Series(front_by_day)
    days = df.index.normalize()
    df = df.copy()
    df["front"] = days.map(fmap)
    cont = df[df["symbol"] == df["front"]].drop(columns=["front"])

    # back-adjust at each roll boundary
    sym_seq = cont["symbol"].values
    change_pts = np.where(sym_seq[1:] != sym_seq[:-1])[0] + 1
    adjust = np.zeros(len(cont))
    cum = 0.0
    # walk backwards: earlier history shifted by sum of later roll gaps
    closes = cont["close"].values
    opens_ = cont["open"].values
    gaps = []
    for cp in change_pts:
        gap = closes[cp] - closes[cp - 1]
        gaps.append((cp, gap))
    offset = np.zeros(len(cont))
    total = 0.0
    for cp, gap in reversed(gaps):
        offset[:cp] += gap
    for col in ("open", "high", "low", "close"):
        cont[col] = cont[col].values + offset
    cont = cont.drop(columns=["symbol"]).sort_index()
    rolls = [(str(cont.index[cp]), round(g, 2)) for cp, g in gaps]
    return cont, rolls


def to_eastern_5min(cont, minutes=5):
    et = cont.tz_convert("America/New_York")
    et.index = et.index.tz_localize(None)
    agg = et.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna(subset=["close"])
    agg = agg[agg["volume"] > 0]
    return agg
