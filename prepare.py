"""
prepare.py — Fixed Evaluation Harness (DO NOT EDIT)
====================================================
The agent CANNOT modify this file. Only strategy.py is editable.

Responsibilities:
  - Scrape current S&P 500 constituents from Wikipedia (cached in DuckDB)
  - Incrementally download 10y daily OHLCV from yfinance (cached in DuckDB)
  - Import strategy.py and run its signals against every S&P 500 ticker
  - Aggregate per-ticker metrics into ONE composite score
  - Print the score to stdout so the agent can parse it

Usage:
    python prepare.py --fetch    # first-time / periodic data refresh
    python prepare.py            # run backtest across the full S&P 500
"""

import argparse
import datetime as dt
import io
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration (FIXED) ────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
DB_PATH         = ROOT / "data" / "sp500.duckdb"
PERIOD_YEARS    = 10
INIT_CAPITAL    = 100_000.0          # USD per ticker (normalized, not portfolio)
FEES_PCT        = 0.0005             # 0.05% per side (realistic US equity)
MIN_TRADES      = 5                  # exclude tickers with fewer trades
MIN_BARS        = 504                # need at least ~2y of data per ticker
TRADING_DAYS    = 252

# Aggregate score targets
TARGET_MEDIAN_SHARPE = 0.8
TARGET_BREADTH       = 0.55          # fraction of tickers with Sharpe > 0

W_SHARPE  = 0.6
W_BREADTH = 0.4

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ── DuckDB setup ─────────────────────────────────────────────────────────────
def get_conn() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker    VARCHAR,
            date      DATE,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            adj_close DOUBLE,
            volume    BIGINT,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS constituents (
            ticker     VARCHAR PRIMARY KEY,
            name       VARCHAR,
            sector     VARCHAR,
            updated_at TIMESTAMP
        );
        CREATE SEQUENCE IF NOT EXISTS iterations_id_seq;
        CREATE TABLE IF NOT EXISTS iterations (
            id              INTEGER PRIMARY KEY DEFAULT nextval('iterations_id_seq'),
            git_sha         VARCHAR,
            ts              TIMESTAMP,
            composite_score DOUBLE,
            median_sharpe   DOUBLE,
            breadth         DOUBLE,
            tickers_used    INTEGER
        );
        CREATE TABLE IF NOT EXISTS backtest_results (
            iteration_id INTEGER,
            ticker       VARCHAR,
            total_return DOUBLE,
            sharpe       DOUBLE,
            max_dd       DOUBLE,
            trades       INTEGER,
            PRIMARY KEY (iteration_id, ticker)
        );
    """)
    return conn


# ── S&P 500 universe ─────────────────────────────────────────────────────────
def scrape_sp500(conn) -> list[str]:
    """Scrape the current S&P 500 constituents from Wikipedia and upsert."""
    print("[prepare.py] Scraping S&P 500 from Wikipedia...", flush=True)
    html = requests.get(
        SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 (sp500-autoresearch)"},
        timeout=30,
    ).text
    tables = pd.read_html(io.StringIO(html))
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    df["updated_at"] = dt.datetime.now()
    conn.execute("DELETE FROM constituents")
    conn.execute("INSERT INTO constituents SELECT ticker, name, sector, updated_at FROM df")
    tickers = sorted(df["ticker"].tolist())
    print(f"[prepare.py] {len(tickers)} constituents recorded.", flush=True)
    return tickers


def load_universe(conn) -> list[str]:
    rows = conn.execute("SELECT ticker FROM constituents ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


# ── Incremental data fetch ───────────────────────────────────────────────────
def fetch_prices(conn, tickers: list[str]) -> None:
    """For each ticker, fetch only the bars we don't already have."""
    today = dt.date.today()
    oldest = today - dt.timedelta(days=PERIOD_YEARS * 365 + 5)
    failures: list[str] = []

    for ticker in tqdm(tickers, desc="fetch"):
        row = conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
        last = row[0] if row else None

        if last is None:
            start = oldest
        elif last >= today - dt.timedelta(days=1):
            continue  # up to date
        else:
            start = last + dt.timedelta(days=1)

        try:
            raw = yf.download(
                ticker,
                start=str(start),
                end=str(today + dt.timedelta(days=1)),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as e:
            failures.append(f"{ticker}: {e}")
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index().rename(columns={
            "Date":      "date",
            "Open":      "open",
            "High":      "high",
            "Low":       "low",
            "Close":     "close",
            "Adj Close": "adj_close",
            "Volume":    "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = ticker
        df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].dropna()
        if df.empty:
            continue

        conn.register("new_rows", df)
        conn.execute("""
            INSERT INTO prices
            SELECT * FROM new_rows
            WHERE (ticker, date) NOT IN (SELECT ticker, date FROM prices)
        """)
        conn.unregister("new_rows")

    if failures:
        print(f"[prepare.py] {len(failures)} fetch failures (first 5): {failures[:5]}",
              flush=True)


def load_prices(conn, ticker: str) -> pd.DataFrame:
    df = conn.execute("""
        SELECT date, open, high, low, close, adj_close, volume
        FROM prices WHERE ticker = ? ORDER BY date
    """, [ticker]).fetch_df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # The agent's strategy uses `close` — serve split/dividend-adjusted.
    df["close"] = df["adj_close"]
    return df[["open", "high", "low", "close", "volume"]]


# ── Per-ticker backtest ──────────────────────────────────────────────────────
def backtest_one(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
                 allocation: float) -> dict:
    """Long-only single-asset backtest. Shifts signals by 1 bar (no lookahead)."""
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    close = df["close"].values
    n = len(close)
    cash = INIT_CAPITAL
    shares = 0.0
    equity = np.empty(n)
    trades = 0

    for i in range(n):
        price = close[i]
        if shares == 0 and entries.iloc[i]:
            spend = cash * allocation
            shares = spend / (price * (1 + FEES_PCT))
            cash -= shares * price * (1 + FEES_PCT)
            trades += 1
        elif shares > 0 and exits.iloc[i]:
            cash += shares * price * (1 - FEES_PCT)
            shares = 0.0
        equity[i] = cash + shares * price

    # close any open position at the final bar for fair accounting
    if shares > 0:
        cash += shares * close[-1] * (1 - FEES_PCT)
        equity[-1] = cash

    eq = pd.Series(equity, index=df.index)
    rets = eq.pct_change().dropna()
    if rets.std() == 0 or len(rets) < 30:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))

    total_return = float((eq.iloc[-1] / INIT_CAPITAL - 1.0) * 100)
    running_max = eq.cummax()
    max_dd = float(((eq - running_max) / running_max).min() * 100)

    return dict(
        total_return=total_return,
        sharpe=sharpe,
        max_dd=abs(max_dd),
        trades=trades,
    )


# ── Aggregate score ──────────────────────────────────────────────────────────
def compute_score(median_sharpe: float, breadth: float) -> float:
    s_miss = max(0.0, 1.0 - median_sharpe / TARGET_MEDIAN_SHARPE)
    b_miss = max(0.0, 1.0 - breadth       / TARGET_BREADTH)
    return W_SHARPE * s_miss + W_BREADTH * b_miss


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_universe_backtest(conn) -> None:
    tickers = load_universe(conn)
    if not tickers:
        sys.exit("[prepare.py] No universe loaded. Run: python prepare.py --fetch")

    # Fresh import of strategy.py every run
    import importlib, sys as _sys
    if "strategy" in _sys.modules:
        del _sys.modules["strategy"]
    import strategy  # noqa

    per_ticker: list[dict] = []
    for ticker in tqdm(tickers, desc="backtest"):
        df = load_prices(conn, ticker)
        if len(df) < MIN_BARS:
            continue
        try:
            entries, exits, allocation = strategy.generate_signals(df)
        except Exception as e:
            print(f"[prepare.py] {ticker}: strategy raised {e}", flush=True)
            continue

        if not isinstance(entries, pd.Series) or not isinstance(exits, pd.Series):
            sys.exit("[prepare.py] generate_signals must return (pd.Series, pd.Series, float)")
        if not (0.0 < float(allocation) <= 1.0):
            sys.exit(f"[prepare.py] allocation must be in (0,1], got {allocation}")

        entries = entries.reindex(df.index).fillna(False).astype(bool)
        exits   = exits.reindex(df.index).fillna(False).astype(bool)

        m = backtest_one(df, entries, exits, float(allocation))
        if m["trades"] < MIN_TRADES:
            continue
        m["ticker"] = ticker
        per_ticker.append(m)

    if not per_ticker:
        print("COMPOSITE SCORE: 1.0000  (no tickers met MIN_TRADES)", flush=True)
        return

    sharpes = np.array([r["sharpe"] for r in per_ticker])
    median_sharpe = float(np.median(sharpes))
    breadth       = float((sharpes > 0).mean())
    score         = compute_score(median_sharpe, breadth)

    # Persist the iteration + per-ticker results
    iter_id = conn.execute("""
        INSERT INTO iterations (git_sha, ts, composite_score, median_sharpe, breadth, tickers_used)
        VALUES (?, ?, ?, ?, ?, ?)
        RETURNING id
    """, [git_sha(), dt.datetime.now(), score, median_sharpe, breadth, len(per_ticker)]
    ).fetchone()[0]

    rows_df = pd.DataFrame([{"iteration_id": iter_id, **r} for r in per_ticker])
    conn.register("rows_df", rows_df)
    conn.execute("""
        INSERT INTO backtest_results
        SELECT iteration_id, ticker, total_return, sharpe, max_dd, trades FROM rows_df
    """)
    conn.unregister("rows_df")

    print("", flush=True)
    print(f"TICKERS SCORED:   {len(per_ticker)}",      flush=True)
    print(f"MEDIAN SHARPE:    {median_sharpe:.4f}",    flush=True)
    print(f"BREADTH (>0):     {breadth:.4f}",          flush=True)
    print(f"COMPOSITE SCORE:  {score:.4f}",            flush=True)


# ── Entrypoint ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Refresh S&P 500 list and incrementally update prices.")
    args = parser.parse_args()

    conn = get_conn()
    if args.fetch:
        tickers = scrape_sp500(conn)
        fetch_prices(conn, tickers)
        print("[prepare.py] Fetch complete.", flush=True)
    else:
        run_universe_backtest(conn)


if __name__ == "__main__":
    main()
