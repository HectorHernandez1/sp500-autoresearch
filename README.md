# sp500-autoresearch

Autonomous quant researcher that iteratively improves a trading strategy
across the entire S&P 500 using Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
loop pattern.

**One strategy. Tested across ~500 stocks. One number to minimize. Forever.**

## The loop

```
  ┌───────────────────────────┐
  │  Claude reads program.md  │
  └────────────┬──────────────┘
               │
  ┌────────────▼──────────────┐
  │  Edit strategy.py         │  ← only file the agent may touch
  └────────────┬──────────────┘
               │
  ┌────────────▼──────────────┐
  │  python prepare.py        │  ← runs backtest across all 500 S&P names
  └────────────┬──────────────┘
               │
  ┌────────────▼──────────────┐
  │  Read COMPOSITE SCORE     │  ← lower is better
  └────────────┬──────────────┘
               │
         ┌─────┴─────┐
    better?         worse?
         │           │
  git commit    git checkout -- strategy.py
         └─────┬─────┘
               ▼
          (loop forever)
```

The universe itself is the cross-validation: a strategy that wins across
500 mostly-uncorrelated names is far harder to overfit than one that wins
on a single ticker.

## Files

| File               | Role                                                       |
| ------------------ | ---------------------------------------------------------- |
| `prepare.py`       | Fixed harness. DO NOT EDIT. Data + backtest + scoring.     |
| `strategy.py`      | The only editable file. Baseline: EMA 10/30 crossover.     |
| `program.md`       | Rules the agent reads each iteration.                      |
| `loop.sh`          | Bash driver: `claude -p` in a while-loop.                  |
| `best_score.txt`   | Current best composite score (committed, versioned).       |
| `environment.yml`  | Conda env spec (python 3.11 + pip deps).                   |
| `requirements.txt` | pip deps (yfinance, duckdb, pandas, numpy, ta, ...).       |

## Storage

Single DuckDB file at `data/sp500.duckdb` (gitignored). Tables:

- `prices` — OHLCV per ticker, keyed on `(ticker, date)`.
- `constituents` — current S&P 500 membership from Wikipedia.
- `iterations` — every backtest run: `git_sha`, timestamp, score, breadth.
- `backtest_results` — per-ticker metrics for every iteration.

Price fetches are **incremental**: `MAX(date)` per ticker is checked, only
missing bars are pulled from yfinance. First fetch is slow (~500 tickers);
every subsequent `--fetch` is fast.

## Score

```
composite = 0.6 × max(0, 1 − median_sharpe / 0.8)
          + 0.4 × max(0, 1 − breadth       / 0.55)
```

- `median_sharpe` — median annualized Sharpe across all tickers scored.
- `breadth` — fraction of tickers with Sharpe > 0.
- **Lower is better.** `0.0` means both targets met.
- Tickers with fewer than 5 trades or less than ~2y of data are excluded.

## Setup

```bash
conda env create -f environment.yml
conda activate sp500-autoresearch

# First-time data pull (~2-5 min, hits yfinance ~500 times).
python prepare.py --fetch

# Baseline score for the starter EMA 10/30 strategy.
python prepare.py
```

## Bootstrap the baseline

After the first `python prepare.py`, record the baseline score as the
initial best:

```bash
echo "0.XXXX" > best_score.txt     # paste the COMPOSITE SCORE printed above
git add best_score.txt
git commit -m "baseline: EMA 10/30 — score 0.XXXX"
```

## Run the loop

```bash
./loop.sh
```

Each iteration = one `claude -p program.md` invocation. The agent reads
`best_score.txt`, edits `strategy.py`, runs `prepare.py`, and either commits
(if the new score is lower) or `git checkout`s (if worse). `Ctrl-C` to stop.

Environment variables:
- `MAX_ITERS` (default `1000`) — max iterations before the script exits.
- `SLEEP_BETWEEN` (default `5`) — seconds between iterations.

## Caveats

- **Survivorship bias.** Uses today's S&P 500 membership against 10y of
  history, so strategies appear better than they would on the actual
  historical universe. Real money needs historical constituent tracking.
- **Overfitting still possible.** 500-name breadth is a strong defense but
  not a guarantee. Long-running loops will eventually find something that
  fits the full 10y window without being a true signal. Walk-forward
  validation is the next hardening step.
- **yfinance is free and flaky.** Rate limits, missing data, and occasional
  corporate-action issues are normal. The harness skips tickers with bad
  data rather than halting.
- **No shorting, no leverage, one asset at a time.** Backtest is long-only
  on a single name with a fixed `allocation` fraction. Cross-sectional or
  portfolio-level strategies are out of scope for v1.

## What to try next

- Walk-forward evaluation (train on rolling window, score out-of-sample).
- Historical S&P constituents (solves survivorship bias).
- Portfolio-level backtest (rank stocks, hold top N, rebalance monthly).
- GPU-accelerated backtests via cuDF once the universe grows (currently CPU
  is more than fast enough).
