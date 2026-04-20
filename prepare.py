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
    python prepare.py            # single-strategy backtest on CPU
    python prepare.py --grid     # GPU parameter sweep over strategy.PARAM_GRID
"""

import argparse
import datetime as dt
import io
import itertools
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

# Walk-forward: split each ticker's return stream into N_FOLDS equal chunks.
# Ticker's Sharpe = median across fold Sharpes (requires ≥ MIN_FOLD_BARS per fold).
# Protects against strategies that only work in one macro regime.
N_FOLDS        = 5
MIN_FOLD_BARS  = 60

# Grid mode
MAX_COMBOS  = 500
GPU_DEVICE  = "cuda"     # falls back to cpu if unavailable
GRID_BEST_W = 0.5        # weight of best-combo score
GRID_MED_W  = 0.5        # weight of median-combo score

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

    # Walk-forward Sharpe: median across N_FOLDS non-overlapping chunks.
    # A strategy must be consistent across time-slices, not just net-positive
    # over 10y that's dominated by one regime.
    sharpe = _walk_forward_sharpe(rets)

    total_return = float((eq.iloc[-1] / INIT_CAPITAL - 1.0) * 100)
    running_max = eq.cummax()
    max_dd = float(((eq - running_max) / running_max).min() * 100)

    return dict(
        total_return=total_return,
        sharpe=sharpe,
        max_dd=abs(max_dd),
        trades=trades,
    )


def _walk_forward_sharpe(rets: pd.Series) -> float:
    """Median Sharpe across N_FOLDS non-overlapping chunks of the return series."""
    n = len(rets)
    if n < N_FOLDS * MIN_FOLD_BARS:
        return 0.0
    fold_size = n // N_FOLDS
    fold_sharpes: list[float] = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end   = n if i == N_FOLDS - 1 else (i + 1) * fold_size
        chunk = rets.iloc[start:end]
        if len(chunk) < MIN_FOLD_BARS or chunk.std() == 0:
            continue
        fold_sharpes.append(float(chunk.mean() / chunk.std() * np.sqrt(TRADING_DAYS)))
    if not fold_sharpes:
        return 0.0
    return float(np.median(fold_sharpes))


# ── Aggregate score ──────────────────────────────────────────────────────────
def compute_score(median_sharpe: float, breadth: float, coverage: float = 1.0) -> float:
    """
    Composite score, lower is better.
    coverage = scored_tickers / eligible_tickers. Shrinking the scored
    universe inflates the score (a penalty) so strategies cannot game the
    MIN_TRADES filter by opting out of hard tickers.
    """
    s_miss = max(0.0, 1.0 - median_sharpe / TARGET_MEDIAN_SHARPE)
    b_miss = max(0.0, 1.0 - breadth       / TARGET_BREADTH)
    raw    = W_SHARPE * s_miss + W_BREADTH * b_miss
    return raw / max(coverage, 0.01)


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
    eligible = 0
    for ticker in tqdm(tickers, desc="backtest"):
        df = load_prices(conn, ticker)
        if len(df) < MIN_BARS:
            continue
        eligible += 1
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
    coverage      = float(len(per_ticker) / max(eligible, 1))
    score         = compute_score(median_sharpe, breadth, coverage)

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
    print(f"TICKERS SCORED:   {len(per_ticker)} / {eligible}", flush=True)
    print(f"COVERAGE:         {coverage:.4f}",         flush=True)
    print(f"MEDIAN SHARPE:    {median_sharpe:.4f}",    flush=True)
    print(f"BREADTH (>0):     {breadth:.4f}",          flush=True)
    print(f"COMPOSITE SCORE:  {score:.4f}",            flush=True)


# ── GPU grid backtest ────────────────────────────────────────────────────────
# NOTE: grid backtest still computes full-period Sharpe (not walk-forward).
# It's not wired into the loop and is kept as an experimental --grid flag.
# If you re-enable it, port _walk_forward_sharpe to torch.
def _get_torch_device():
    """Import torch lazily and return a device. Falls back to cpu."""
    import torch
    if GPU_DEVICE == "cuda" and torch.cuda.is_available():
        return torch, torch.device("cuda:0"), torch.cuda.get_device_name(0)
    return torch, torch.device("cpu"), "cpu"


def backtest_grid_one_ticker(torch, device,
                              close_np: np.ndarray,
                              entries_np: np.ndarray,   # (n_combos, n_bars) bool
                              exits_np:   np.ndarray,   # (n_combos, n_bars) bool
                              allocation: float) -> dict:
    """
    Vectorized per-bar long-only backtest across all combos for ONE ticker.
    Returns per-combo sharpe, total_return, max_dd, trades as numpy arrays.
    """
    close   = torch.from_numpy(close_np.astype(np.float32)).to(device)
    entries = torch.from_numpy(entries_np).to(device)
    exits   = torch.from_numpy(exits_np).to(device)

    n_combos, n_bars = entries.shape
    shares = torch.zeros(n_combos, device=device)
    cash   = torch.full((n_combos,), INIT_CAPITAL, device=device)
    equity = torch.zeros((n_combos, n_bars), device=device)
    trades = torch.zeros(n_combos, device=device, dtype=torch.int32)

    fees_in  = 1.0 + FEES_PCT
    fees_out = 1.0 - FEES_PCT
    zero     = torch.zeros_like(shares)

    for i in range(n_bars):
        price = close[i]
        # Entry: flat AND entry signal
        can_enter = (shares == 0) & entries[:, i]
        new_shares = (cash * allocation) / (price * fees_in)
        shares = torch.where(can_enter, new_shares, shares)
        cash   = torch.where(can_enter, cash - new_shares * price * fees_in, cash)
        trades = trades + can_enter.to(torch.int32)

        # Exit: long AND exit signal
        can_exit = (shares > 0) & exits[:, i]
        cash   = torch.where(can_exit, cash + shares * price * fees_out, cash)
        shares = torch.where(can_exit, zero, shares)

        equity[:, i] = cash + shares * price

    # Close remaining at last bar
    last_price = close[-1]
    cash   = cash + shares * last_price * fees_out
    equity[:, -1] = cash

    rets     = equity[:, 1:] / equity[:, :-1] - 1.0
    mean_ret = rets.mean(dim=1)
    std_ret  = rets.std(dim=1)
    sharpe   = torch.where(
        std_ret > 0,
        mean_ret / std_ret * (TRADING_DAYS ** 0.5),
        torch.zeros_like(mean_ret),
    )
    total_return = (equity[:, -1] / INIT_CAPITAL - 1.0) * 100.0
    cummax = torch.cummax(equity, dim=1).values
    max_dd = ((equity - cummax) / cummax).min(dim=1).values.abs() * 100.0

    return dict(
        sharpe       = sharpe.cpu().numpy(),
        total_return = total_return.cpu().numpy(),
        max_dd       = max_dd.cpu().numpy(),
        trades       = trades.cpu().numpy(),
    )


def _enumerate_grid(grid: dict) -> list[dict]:
    keys   = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_universe_backtest_grid(conn) -> None:
    tickers = load_universe(conn)
    if not tickers:
        sys.exit("[prepare.py] No universe loaded. Run: python prepare.py --fetch")

    import importlib, sys as _sys
    if "strategy" in _sys.modules:
        del _sys.modules["strategy"]
    import strategy  # noqa

    if not hasattr(strategy, "PARAM_GRID") or not isinstance(strategy.PARAM_GRID, dict):
        sys.exit("[prepare.py] strategy.PARAM_GRID missing — grid mode requires it.")

    combos = _enumerate_grid(strategy.PARAM_GRID)
    n_combos = len(combos)
    if n_combos == 0:
        sys.exit("[prepare.py] PARAM_GRID is empty.")
    if n_combos > MAX_COMBOS:
        sys.exit(f"[prepare.py] grid has {n_combos} combos (cap {MAX_COMBOS}); shrink it.")

    torch, device, device_name = _get_torch_device()
    print(f"[prepare.py] grid mode: {n_combos} combos × S&P 500, device={device_name}",
          flush=True)

    # (n_combos, n_tickers_eligible) accumulators
    sharpe_cols: list[np.ndarray] = []
    trades_cols: list[np.ndarray] = []
    scored_tickers: list[str] = []
    eligible = 0
    ticker_allocation = None

    for ticker in tqdm(tickers, desc="grid"):
        df = load_prices(conn, ticker)
        if len(df) < MIN_BARS:
            continue
        eligible += 1

        entries_list = []
        exits_list   = []
        ok = True
        for params in combos:
            try:
                e, x, alloc = strategy.generate_signals(df, params)
            except Exception as ex:
                ok = False
                print(f"[prepare.py] {ticker}: strategy raised {ex}", flush=True)
                break
            if not isinstance(e, pd.Series) or not isinstance(x, pd.Series):
                sys.exit("[prepare.py] generate_signals must return (pd.Series, pd.Series, float)")
            e = e.reindex(df.index).fillna(False).astype(bool).shift(1).fillna(False).astype(bool)
            x = x.reindex(df.index).fillna(False).astype(bool).shift(1).fillna(False).astype(bool)
            entries_list.append(e.values)
            exits_list.append(x.values)
            ticker_allocation = float(alloc)
        if not ok:
            continue

        entries_arr = np.stack(entries_list)   # (n_combos, n_bars)
        exits_arr   = np.stack(exits_list)
        m = backtest_grid_one_ticker(
            torch, device, df["close"].values, entries_arr, exits_arr, ticker_allocation,
        )
        sharpe_cols.append(m["sharpe"])
        trades_cols.append(m["trades"])
        scored_tickers.append(ticker)

    if not scored_tickers:
        print("COMPOSITE SCORE: 1.0000  (no tickers scored)", flush=True)
        return

    sharpe_mat = np.stack(sharpe_cols, axis=1)   # (n_combos, n_tickers)
    trades_mat = np.stack(trades_cols, axis=1)

    per_combo_score = np.full(n_combos, 1.0)
    per_combo_median_sharpe = np.zeros(n_combos)
    per_combo_breadth       = np.zeros(n_combos)
    per_combo_coverage      = np.zeros(n_combos)

    for c in range(n_combos):
        mask = trades_mat[c] >= MIN_TRADES
        if mask.sum() == 0:
            continue
        sh = sharpe_mat[c, mask]
        ms = float(np.median(sh))
        br = float((sh > 0).mean())
        cov = float(mask.sum() / max(eligible, 1))
        per_combo_score[c]         = compute_score(ms, br, cov)
        per_combo_median_sharpe[c] = ms
        per_combo_breadth[c]       = br
        per_combo_coverage[c]      = cov

    best_idx     = int(per_combo_score.argmin())
    best_score   = float(per_combo_score[best_idx])
    median_score = float(np.median(per_combo_score))
    stability    = 1.0 - (median_score - best_score) / max(median_score, 0.01)
    final_score  = GRID_BEST_W * best_score + GRID_MED_W * median_score
    best_params  = combos[best_idx]

    # Persist
    iter_id = conn.execute("""
        INSERT INTO iterations (git_sha, ts, composite_score, median_sharpe, breadth, tickers_used)
        VALUES (?, ?, ?, ?, ?, ?)
        RETURNING id
    """, [git_sha(), dt.datetime.now(), final_score,
          float(per_combo_median_sharpe[best_idx]),
          float(per_combo_breadth[best_idx]),
          int((trades_mat[best_idx] >= MIN_TRADES).sum())]
    ).fetchone()[0]

    best_rows = pd.DataFrame({
        "iteration_id": iter_id,
        "ticker": scored_tickers,
        "total_return": 0.0,                       # not tracked per-combo here
        "sharpe": sharpe_mat[best_idx],
        "max_dd": 0.0,
        "trades": trades_mat[best_idx],
    })
    conn.register("best_rows", best_rows)
    conn.execute("""
        INSERT INTO backtest_results
        SELECT iteration_id, ticker, total_return, sharpe, max_dd, trades FROM best_rows
    """)
    conn.unregister("best_rows")

    print("", flush=True)
    print(f"COMBOS EVALUATED: {n_combos}",         flush=True)
    print(f"TICKERS ELIGIBLE: {eligible}",         flush=True)
    print(f"BEST PARAMS:      {best_params}",      flush=True)
    print(f"BEST SCORE:       {best_score:.4f} "
          f"(median sharpe {per_combo_median_sharpe[best_idx]:.4f}, "
          f"breadth {per_combo_breadth[best_idx]:.4f}, "
          f"coverage {per_combo_coverage[best_idx]:.4f})", flush=True)
    print(f"MEDIAN SCORE:     {median_score:.4f}", flush=True)
    print(f"STABILITY:        {stability:.4f}",    flush=True)
    print(f"COMPOSITE SCORE:  {final_score:.4f}",  flush=True)


# ── Entrypoint ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Refresh S&P 500 list and incrementally update prices.")
    parser.add_argument("--grid", action="store_true",
                        help="GPU parameter sweep over strategy.PARAM_GRID.")
    args = parser.parse_args()

    conn = get_conn()
    if args.fetch:
        tickers = scrape_sp500(conn)
        fetch_prices(conn, tickers)
        print("[prepare.py] Fetch complete.", flush=True)
    elif args.grid:
        run_universe_backtest_grid(conn)
    else:
        run_universe_backtest(conn)


if __name__ == "__main__":
    main()
