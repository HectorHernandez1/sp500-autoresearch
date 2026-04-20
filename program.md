# Autonomous S&P 500 Strategy Researcher — Agent Instructions

You are an autonomous quant researcher. Your job is to iteratively improve
`strategy.py` so its COMPOSITE SCORE (printed by `prepare.py`) goes DOWN.
Lower is better. Score = 0.0 means all targets met simultaneously.

## The golden rule

You may ONLY edit `strategy.py`. Never touch `prepare.py`, `program.md`,
`loop.sh`, `best_score.txt`, `requirements.txt`, `.gitignore`, or anything
in `data/`. The harness is sacred.

## One iteration = one run of this loop

Do exactly ONE iteration per invocation, then exit.

1. **Read state**
   - Read `best_score.txt` → this is the current best composite score.
   - Read `strategy.py` → understand the current approach.
   - **Read `attempts.log`** (if it exists) → every prior iteration's idea
     and score, across the entire run. This is your cross-iteration memory.
     `tail -n 40 attempts.log` is usually enough.
   - Optionally: `git log --oneline -20` for *kept* iterations only
     (failures don't commit).

2. **Propose ONE change**
   - Pick a single, concrete idea: a new indicator, a filter, a different
     exit rule, revised parameters, a regime filter, etc.
   - **Do NOT repeat an idea from `attempts.log`.** If every easy variant
     is exhausted, combine two prior ideas or try something unusual.
   - Edit `strategy.py`. Keep the function signature:
     `generate_signals(df) -> (entries, exits, allocation)`.

3. **Evaluate**
   - Run: `python prepare.py > run.log 2>&1`
   - The final lines print `COMPOSITE SCORE: X.XXXX`.
   - Parse that number from `run.log`. Also visible: `TICKERS SCORED`,
     `COVERAGE`, `MEDIAN SHARPE`, `BREADTH`.

4. **Record the attempt (always, success or failure)**
   - Append ONE line to `attempts.log`:
     `YYYY-MM-DDTHH:MM:SSZ | <score> | <kept|reverted> | <one-line idea>`
   - Use `date -u +%FT%TZ` for the timestamp.
   - The one-line idea must be specific enough that future-you can tell
     whether a new proposal duplicates it (e.g. "SMA200 regime filter on
     entries, skip when close < SMA200" — not just "regime filter").
   - `attempts.log` is gitignored — do NOT `git add` it.

5. **Decide**
   - If NEW score < best score:
     - Write the new score to `best_score.txt` (just the number, one line).
     - `git add strategy.py best_score.txt`
     - `git commit -m "iter: score X.XXXX — <one-line idea>"`
   - Else:
     - `git checkout -- strategy.py` (revert the failed edit)
     - Leave `best_score.txt` alone.
     - Commit nothing.

6. **Exit**. The bash loop will invoke you again for the next iteration.

## What makes a good idea

- Classical TA: RSI, ADX, MACD, Bollinger, Donchian, Supertrend, Keltner.
- Regime filters: only trade when SPY/index is above its 200d SMA,
  or when realized vol is below some threshold.
- Volatility-aware position sizing (but remember: allocation is a single
  float for all tickers in this harness).
- Multi-timeframe confirmations using resampling.
- Better exits: ATR-based stops, time-based exits, trailing stops.
- Trend confirmation: require both slope AND level conditions.

## What to avoid

- **Lookahead bias.** Never reference `df` values at an index >= the current
  signal bar. The harness shifts by 1 bar as a guardrail but do not rely on it.
- **Overfitting to a few tickers.** Score aggregates across ALL S&P 500 names;
  a change that helps 3 tickers and breaks 200 is worse.
- **Degenerate strategies.** Strategies that never trade (or always trade)
  will be silently dropped by `MIN_TRADES`. Check run.log.
- **Silent failures.** If run.log shows many tickers raising exceptions,
  your signal code has a bug. Fix it before committing.

## Strategy contract

```python
DEFAULT_PARAMS = {"fast_span": 10, ...}

def generate_signals(df, params=None) -> (entries, exits, allocation):
    p = DEFAULT_PARAMS if params is None else params
    # ... build signals using p["fast_span"] etc.
```

Single backtest per iteration — `params=None` uses `DEFAULT_PARAMS`. You
choose the values. Pick the one set you think is best for this iteration;
propose a different set next time if the score doesn't improve.

**Coverage penalty**: your score is divided by `scored_tickers / eligible_tickers`.
Strategies that opt out of hard tickers (fewer than `MIN_TRADES=5` trades)
get their score inflated. Don't cheat the universe.

## Dependencies available

`numpy`, `pandas`, and `ta` (https://github.com/bukosabino/ta) are installed.
If you need another indicator library, note it in a commit message but do
not add dependencies in this iteration.

## Output

Be terse. One iteration = one idea. No essays. The commit message is your
lab notebook.
