# Autonomous S&P 500 Strategy Researcher — Agent Instructions

You are an autonomous quant researcher. Your job is to iteratively improve
`strategy.py` so its COMPOSITE SCORE (printed by `prepare.py`) goes DOWN.
Lower is better. Score = 0.0 means all targets met simultaneously.

## The golden rule

You may ONLY edit `strategy.py`. Never touch `prepare.py`, `program.md`,
`loop.sh`, `best_score.txt`, `requirements.txt`, `.gitignore`, or anything
in `data/`. The harness is sacred.

## One iteration = one run of this loop

Do exactly ONE iteration per invocation, then exit. `loop.sh` has already
checked out an `experiments/autoloop-*` branch — do NOT switch branches
and do NOT push. Your commits stay local on the experiment branch until
the user decides to merge winners into main.

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

Score now uses **walk-forward Sharpe** (median across 5 non-overlapping
time-folds per ticker, `MIN_FOLD_BARS=60`). A strategy must be consistent
across time-slices, not just net-positive over the whole 10y window —
regime-dependent wins are punished.

### Idea bank (check `attempts.log` first, don't repeat)

**Entry alternatives**
- Donchian breakout on `n`-day high (n∈{20,55}); pair with volatility filter.
- Close above rolling `k*ATR` band above short MA (Keltner channel breakout).
- Z-score mean-reversion: enter long when close is `<-1.5σ` below a 20d MA
  (contrarian — opposite of the current trend-following bias).
- MACD bullish cross with histogram > 0.
- ADX > 25 gate on the current EMA cross (require *trending* market).

**Exit alternatives**
- Parabolic SAR instead of Chandelier.
- ATR trailing stop from the entry price (not the rolling high).
- Fixed fractional profit target AND stop (e.g. +2R/-1R).
- Chandelier + time stop (exit after N bars regardless).
- Exit on RSI > 75 (take profit into momentum blow-off).

**Regime / filter layers**
- Only trade when SPY 200d slope is rising (compute SPY separately with `yfinance`).
- Realized-vol regime: take trades only when 20d realized vol is below
  its trailing 252d median.
- VIX proxy: skip entries when (SPY high-low range / SPY close) is elevated.
- Earnings blackout: require N bars since last price gap > 5% (proxy).
- Trend strength: require 50d SMA > 200d SMA AND slope(50d) > 0.

**Multi-timeframe**
- Resample to weekly; require weekly EMA cross aligned with daily entries.
- Monthly trend gate: only enter if last month's close > close from 6 months ago.

**Volatility-adaptive**
- Scale `allocation` proportional to `1/realized_vol` (but cap at 1.0).
- Wider Chandelier stop when ATR is high (vol-adaptive multiplier).
- Skip entries when ATR exceeds a fraction of price (too noisy).

**Cross-signal confirmation**
- EMA cross + price above upper Bollinger band at entry.
- Supertrend + OBV rising over last N bars.
- Dual-EMA cross (both 10/30 AND 50/200 must agree).

**Signal quality / timing**
- Pullback entry: EMA cross happened in last 5 bars AND today's close is
  within 1 ATR of the 20d high (buy the dip inside the trend).
- N-bar breakout confirmation: only enter when close > prior `k` bars' high.
- Entry only on "narrow range" days (today's range < last 7d avg range).

**Unconventional**
- Day-of-week filter (e.g. skip Mondays).
- Consecutive up/down day count (streaks).
- Two-legged confirmation across Tue close *and* Thu close.

Pick ONE idea, combine intelligently with what's already working — do NOT
strip out the existing Chandelier stop unless you're deliberately replacing it.

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
