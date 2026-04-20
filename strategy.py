"""
strategy.py — The ONLY file the agent is allowed to modify.
============================================================
Rules:
  - The function signature MUST stay: generate_signals(df) -> (entries, exits, allocation)
  - `entries` and `exits` must be boolean pd.Series aligned to df.index
  - `allocation` must be a float in (0, 1]
  - Do NOT look at future bars — the harness shifts signals by 1 bar as a safety net,
    but you should still build signals out of past-only information
  - You may add helper functions, imports, and parameters freely

Available on df (DatetimeIndex): open, high, low, close, volume
`close` is split/dividend-adjusted already.

EMA 10/30 crossover entries + Chandelier (ATR trailing) stop exits.
"""

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

FAST_SPAN  = 10
SLOW_SPAN  = 30
ATR_LEN    = 22
ATR_MULT   = 3.0
ALLOCATION = 0.95


def generate_signals(df: pd.DataFrame):
    fast = df["close"].ewm(span=FAST_SPAN, adjust=False).mean()
    slow = df["close"].ewm(span=SLOW_SPAN, adjust=False).mean()

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_LEN
    ).average_true_range()
    rolling_high = df["close"].rolling(ATR_LEN, min_periods=ATR_LEN).max()
    chandelier_stop = rolling_high - ATR_MULT * atr
    stop_hit = df["close"] < chandelier_stop

    entries = cross_up.fillna(False).astype(bool)
    exits   = (cross_dn | stop_hit).fillna(False).astype(bool)
    return entries, exits, ALLOCATION
