# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The critical rule

This repo is an **autonomous research loop**, not a normal codebase. When invoked via `loop.sh`, Claude reads `program.md` and gets exactly ONE job: edit `strategy.py` to lower the COMPOSITE SCORE printed by `prepare.py`.

**Inside the research loop, `strategy.py` is the ONLY editable file.** `prepare.py`, `program.md`, `loop.sh`, `best_score.txt`, `requirements.txt`, `environment.yml`, `.gitignore`, and `data/` are off-limits — the harness is sacred and changing it invalidates every prior score. See `program.md` for the full agent protocol.

Outside the loop (e.g., a user asking you to fix a harness bug or add a feature), normal editing applies — but flag any change to the harness files above as score-invalidating so the user can reset `best_score.txt` accordingly.

## Commands

```bash
conda env create -f environment.yml         # first-time env setup
conda activate sp500-autoresearch

python prepare.py --fetch                   # scrape S&P 500 + incremental yfinance pull (slow first time)
python prepare.py                           # run full backtest, print COMPOSITE SCORE
./loop.sh                                   # autonomous loop — one `claude -p program.md` per iteration
```

Env vars for `loop.sh`: `MAX_ITERS` (default 1000), `SLEEP_BETWEEN` (default 5s).

There are no unit tests, no linter, no build step. The backtest *is* the test.

## Architecture

Three files carry all the meaning; the rest is plumbing:

- **[strategy.py](strategy.py)** — contract: `generate_signals(df) -> (entries, exits, allocation)`. `entries`/`exits` are boolean `pd.Series` aligned to `df.index`; `allocation` is a float in `(0, 1]`. `df` has a DatetimeIndex and columns `open/high/low/close/volume`, where `close` is already split/dividend-adjusted.
- **[prepare.py](prepare.py)** — fixed harness. Loads the universe from DuckDB, imports `strategy.py` fresh each run (busts the module cache), runs [backtest_one()](prepare.py#L208) against every ticker with ≥504 bars, filters out tickers with fewer than `MIN_TRADES=5` trades, aggregates into a composite score.
- **[program.md](program.md)** — the agent's system prompt. Read this before making any change that touches the loop's behavior.

### Scoring (in [prepare.py:258](prepare.py#L258))

```
composite = 0.6 × max(0, 1 − median_sharpe / 0.8)
          + 0.4 × max(0, 1 − breadth       / 0.55)
```

Lower is better; 0.0 means both targets met. `breadth` = fraction of scored tickers with Sharpe > 0. Universe breadth substitutes for cross-validation: a strategy that wins across ~500 mostly-uncorrelated names is harder to overfit than one tuned on a single ticker.

### Data layer

Single DuckDB file at `data/sp500.duckdb` (gitignored). Tables:
- `prices` — OHLCV keyed on `(ticker, date)`, upserted incrementally (only fetch bars newer than `MAX(date)` per ticker).
- `constituents` — current S&P 500 from Wikipedia.
- `iterations` + `backtest_results` — every run is persisted with `git_sha`, score, and per-ticker metrics, so you can query historical attempts directly.

### Lookahead safety

[backtest_one()](prepare.py#L211-L212) shifts both signals by 1 bar as a guardrail, but **strategies must still build signals from past-only information**. The shift is a safety net, not a license. Strategies that silently never/always trade get dropped by `MIN_TRADES` — if `run.log` shows the tickers-scored count collapse, that's the cause.

## Known caveats (documented in README)

- **Survivorship bias**: uses today's S&P 500 membership against 10y of history.
- **yfinance is flaky**: rate limits and missing data are expected; the harness logs failures and continues rather than halting.
- **Long-only, single-asset, fixed allocation** per ticker. No shorting, leverage, or cross-sectional strategies in v1.
