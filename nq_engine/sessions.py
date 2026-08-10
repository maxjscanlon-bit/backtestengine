"""Session boundary and RTH helpers.

Two conventions matter here and both were getting silently mishandled:

1. A Globex session runs 18:00 ET to 17:00 ET the following day, with a
   maintenance break 17:00-18:00. Grouping P&L by calendar date splits the
   Sunday and overnight portion away from the RTH portion it belongs to,
   which fabricates extra low-activity "days" and distorts daily Sharpe.

2. Bars are right-labeled: a bar stamped 09:35 covers (09:30, 09:35]. So the
   bar stamped 09:30 covers (09:25, 09:30], which is pre-open, and the bar
   stamped 16:00 covers (15:55, 16:00], which is the last RTH bar. A naive
   `>= 09:30 and < 16:00` filter includes a pre-open bar and drops a real one.
"""

import numpy as np
import pandas as pd

SESSION_START_MIN = 18 * 60      # 18:00 ET, exclusive for right-labeled bars
RTH_OPEN_MIN = 9 * 60 + 30       # 09:30 ET, exclusive
RTH_CLOSE_MIN = 16 * 60          # 16:00 ET, inclusive


def _minute_of_day(index):
    return index.hour * 60 + index.minute


def session_date(index):
    """Trading session date for each right-labeled bar.

    Bars stamped after 18:00 ET belong to the NEXT calendar day's session,
    matching CME trade-date convention.
    """
    mod = _minute_of_day(index)
    base = pd.DatetimeIndex(index).normalize()
    roll = pd.to_timedelta((mod > SESSION_START_MIN).astype(int), unit="D")
    return (base + roll).date


def session_group(pnl):
    """Group a P&L series by trading session rather than calendar date."""
    return pnl.groupby(session_date(pnl.index))


def rth_mask(index):
    """True for bars whose coverage window lies inside the 09:30-16:00 cash session.

    Right-labeled bars: keep (09:30, 16:00], i.e. the 09:35 bar through the
    16:00 bar inclusive.
    """
    mod = _minute_of_day(index)
    weekday = pd.DatetimeIndex(index).dayofweek < 5
    return (mod > RTH_OPEN_MIN) & (mod <= RTH_CLOSE_MIN) & weekday
