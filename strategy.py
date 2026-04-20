"""
strategy.py — The ONLY file the agent is allowed to modify.
============================================================
Rules:
  - Function signature: generate_signals(df, params=None) -> (entries, exits, allocation)
  - `entries` and `exits` must be boolean pd.Series aligned to df.index
  - `allocation` must be a float in (0, 1]
  - `params` is a dict — when None, use DEFAULT_PARAMS (single mode).
    When passed (grid mode), pull every tunable from params.
  - Do NOT look at future bars. The harness shifts signals by 1 bar as a
    safety net, but you should still build signals from past-only info.

Available on df (DatetimeIndex): open, high, low, close, volume
`close` is split/dividend-adjusted already.

EMA fast/slow crossover entries; Chandelier (ATR trailing) stop as sole exit.

PARAM_GRID is used by `python prepare.py --grid` — the harness enumerates
the Cartesian product, evaluates every combo on the GPU, and reports the
best / median / stability across the family.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


DEFAULT_PARAMS = {
    "fast_span":  10,
    "slow_span":  30,
    "atr_len":    22,
    "atr_mult":   3.0,
    "slope_look": 5,
    "rsi_len":    14,
    "rsi_tp":     75,
    "profit_r":   4.0,
}

PARAM_GRID = {
    "fast_span":  [5, 8, 10, 12, 15],
    "slow_span":  [20, 30, 40, 50],
    "atr_len":    [14, 22, 30],
    "atr_mult":   [2.0, 2.5, 3.0, 3.5],
    "slope_look": [3, 5, 10],
}

ALLOCATION = 0.95


def generate_signals(df: pd.DataFrame, params: dict | None = None):
    p = DEFAULT_PARAMS if params is None else params

    fast = df["close"].ewm(span=int(p["fast_span"]), adjust=False).mean()
    slow = df["close"].ewm(span=int(p["slow_span"]), adjust=False).mean()
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    slow_rising = slow > slow.shift(int(p["slope_look"]))
    entries = (cross_up & slow_rising).fillna(False).astype(bool)

    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"],
        window=int(p["atr_len"]),
    ).average_true_range()
    rolling_high = df["high"].rolling(
        int(p["atr_len"]), min_periods=int(p["atr_len"])
    ).max()
    atr_median = atr.rolling(252, min_periods=60).median()
    vol_ratio = (atr / atr_median).clip(lower=1.0, upper=1.5).fillna(1.0)
    dynamic_mult = float(p["atr_mult"]) * vol_ratio
    chandelier_stop = rolling_high - dynamic_mult * atr
    below_stop = df["close"] < chandelier_stop
    stop_hit = below_stop & below_stop.shift(1).fillna(False)

    rsi = RSIIndicator(close=df["close"], window=int(p["rsi_len"])).rsi()
    rsi_tp = rsi > float(p["rsi_tp"])

    entry_price = pd.Series(np.where(entries, df["close"], np.nan), index=df.index).ffill()
    atr_at_entry = pd.Series(np.where(entries, atr, np.nan), index=df.index).ffill()
    profit_level = entry_price + float(p["profit_r"]) * float(p["atr_mult"]) * atr_at_entry
    profit_target = (df["close"] > profit_level).fillna(False)

    exits = (stop_hit | rsi_tp | profit_target).fillna(False).astype(bool)
    return entries, exits, ALLOCATION
