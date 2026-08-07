"""Data loading, integrity checks, and TRAIN/VAL/HOLDOUT splitting for NQ 5-min bars.

Expected input: NinjaTrader continuous futures export, CSV or semicolonless text.
Typical NT8 format: yyyyMMdd HHmmss;open;high;low;close;volume  (Last bars export)
We auto-detect delimiter and column layout.
"""

import pandas as pd
import numpy as np
from pathlib import Path

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_bars(path, tz="America/New_York"):
    """Load NT8 export into a clean OHLCV DataFrame indexed by timestamp."""
    path = Path(path)
    raw = pd.read_csv(path, sep=None, engine="python", header=None)
    if raw.shape[1] < 6:
        raise ValueError(f"Expected >=6 columns, got {raw.shape[1]}")
    # If first row looks like a header, reload with header
    first = str(raw.iloc[0, 0]).lower()
    if any(k in first for k in ("time", "date")):
        raw = pd.read_csv(path, sep=None, engine="python")
        raw.columns = [c.strip().lower() for c in raw.columns]
    else:
        raw = raw.iloc[:, :6]
        raw.columns = REQUIRED_COLS

    if "timestamp" not in raw.columns:
        # assume first col is datetime
        raw = raw.rename(columns={raw.columns[0]: "timestamp"})

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed")
    df = raw.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def integrity_report(df, bar_minutes=5):
    """Run integrity checks. Returns dict of findings. Nothing proceeds if fatal."""
    rep = {}
    rep["rows"] = len(df)
    rep["start"] = str(df.index.min())
    rep["end"] = str(df.index.max())
    rep["duplicate_timestamps"] = int(df.index.duplicated().sum())
    rep["nan_rows"] = int(df.isna().any(axis=1).sum())
    bad_ohlc = ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
                (df["low"] > df[["open", "close", "high"]].min(axis=1)))
    rep["impossible_bars"] = int(bad_ohlc.sum())
    rep["zero_volume_bars"] = int((df["volume"] == 0).sum())
    # bar-to-bar close jumps, flag > 2 percent as possible bad roll
    ret = df["close"].pct_change().abs()
    rep["jumps_gt_2pct"] = int((ret > 0.02).sum())
    rep["max_abs_jump_pct"] = round(float(ret.max()) * 100, 3)
    # session coverage per day
    per_day = df.groupby(df.index.date).size()
    rep["days"] = len(per_day)
    rep["median_bars_per_day"] = int(per_day.median())
    rep["days_lt_half_median"] = int((per_day < per_day.median() / 2).sum())
    rep["fatal"] = bool(rep["duplicate_timestamps"] or rep["impossible_bars"] or rep["nan_rows"])
    return rep


def split_and_lock(df, out_dir, train=0.6, val=0.2):
    """Chronological TRAIN/VAL/HOLDOUT split written to separate files. Holdout untouched after this."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    parts = {
        "train": df.iloc[:i1],
        "val": df.iloc[i1:i2],
        "holdout": df.iloc[i2:],
    }
    for name, part in parts.items():
        part.to_parquet(out / f"{name}.parquet")
    bounds = {k: (str(v.index.min()), str(v.index.max()), len(v)) for k, v in parts.items()}
    return bounds


def load_split(out_dir, name):
    return pd.read_parquet(Path(out_dir) / f"{name}.parquet")
