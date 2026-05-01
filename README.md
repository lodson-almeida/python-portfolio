# python-portfolio
Production-grade Python samples from a Quantitative System Project
# Algorithmic Forex Trading System — Selected Modules

Four production modules extracted from a live algorithmic trading system built in Python. The system runs multi-currency backtests, performs Walk-Forward Analysis (WFA) to validate strategy robustness, executes live trades via the OANDA API (REST, STREAM), and produces detailed performance reports in Excel.

These files cannot be run in isolation — they are components of the larger system and depend on shared data pipelines, configuration objects, and market data files. They are shared here to demonstrate architecture and code quality.

---

## Modules

### `gen_engine.py` — Generic Backtesting Engine

The core execution engine for running parameter-optimisation backtests across multiple currency pairs in parallel.

**What it does:**
- Loads multi-timeframe OHLCV market data (Parquet / Pickle) with UTC timezone enforcement
- Accepts any trading strategy via a pluggable interface — `prepare_data` and `apply_signals` are injected at runtime, making the engine strategy-agnostic
- Iterates over thousands of parameter combinations per currency pair, simulating each and collecting trade results
- Runs pairs in parallel using `tqdm.contrib.concurrent.process_map` with configurable worker count
- Implements a **checkpoint/resume system**: saves progress to disk at configurable intervals so long runs survive crashes or interruptions without losing work
- Aggregates per-pair Parquet results into a single combined dataset and triggers downstream Excel reporting

**Key technical highlights:**
- Multiprocessing-safe design — config objects are serialised into plain dicts before being passed to worker processes
- Explicit `float32` casting throughout to reduce memory footprint when handling millions of trade records
- `gc.collect()` called at strategic points to prevent memory accumulation across hundreds of pairs
- Full `typing` annotations (`Optional`, `Tuple`, `Dict`, `List`, `Any`) on all public functions

---

### `walk_forward_analysis.py` — Walk-Forward Analysis Orchestrator

Implements Walk-Forward Analysis, a standard technique in quantitative finance for testing whether a strategy's optimised parameters generalise to unseen data.

**What it does:**
- Generates rolling In-Sample / Out-of-Sample period windows from a date range (two methods supported: *Loddy* and *Original*)
- For each period: runs the backtesting engine on the In-Sample window, selects the best parameter sets via a filter/ranking step, then re-runs those exact parameters on the Out-of-Sample window
- Aggregates cross-period results and runs parameter frequency analysis — identifying which parameters appear consistently across periods (a robustness signal)
- Handles intelligent **parameter sampling**: given a large parameter space, it loads existing checkpoints to identify already-processed combinations and fills the sample budget with unprocessed ones, ensuring no redundant computation
- Manages the full folder structure (`checkpoints/`, `final_results/`, `trades/`) for each period automatically

**Key technical highlights:**
- Memory-conscious large-dataset handling — intermediate data structures (validated combination lists, processed-params sets) are explicitly cleared with `gc.collect()` after use
- Supports reproducible runs via an optional `RANDOM_SEED` on the sampler
- Conditional logging: switches between a multiprocessing-safe `LogWrapper` (file-per-process) and a standard logger depending on a `DEBUG_LOGGING` flag
- Clean separation of concerns: period generation, folder management, in-sample runs, out-of-sample runs, result aggregation, and parameter analysis are each in their own function

---

### `retry_mechanism.py` — Exponential Backoff Retry

A reusable retry utility for wrapping unreliable network operations (live API calls, price stream reconnections).

**What it does:**
- Implements **exponential backoff with jitter**: wait time doubles each attempt up to a configured ceiling, with random jitter added to prevent thundering-herd reconnection storms
- Configurable maximum failure count with two modes: *reset* (keep retrying indefinitely, cycling the counter) or *stop* (raise the exception after max failures)
- Generic `execute_with_retry(operation, *args, **kwargs)` wrapper accepts any callable, keeping retry logic completely decoupled from business logic

**Key technical highlights:**
- 48 lines, no dependencies beyond the standard library (`random`, `time`)
- Used system-wide: the live trading bot, price streamer, and API layer all share this single class

---

### `gen_performance_metrics.py` — Performance Metrics Engine

Calculates the full suite of trading performance statistics from a DataFrame of completed trades. Used after every backtest run to evaluate each parameter combination.

**What it does:**
- **`UnaffectedMetrics`** — trade counts, win/loss rates, directional breakdown (long vs short), largest/average trade values, and max consecutive win/loss streak (calculated via cumulative-sum vectorisation, not a Python loop)
- **`ScoreAndPipsMetrics`** — score-based P&L (normalised to a $100 account for cross-pair comparability), gross/net pips, profit factor, and CAGR
- **`DollarValueMetrics`** — absolute dollar P&L and dollar-based profit factor
- **`AccountMetrics`** — account equity curve, log returns, Sharpe ratio (annualised), max drawdown, win/loss return ratio
- **`AllMetrics`** — composes all four classes and exposes a single `get_metrics_dict()` that returns every metric as a flat dictionary, ready to be written as a row in the results Parquet file

**Key technical highlights:**
- Strict `float32` usage throughout — running `AllMetrics` on millions of trade rows across thousands of combinations adds up; halving float precision meaningfully reduces memory and speeds up aggregation
- Each sub-class has granular `try/except` blocks with safe defaults, so a single bad trade row cannot abort an entire multi-hour backtest run
- No loops — streak calculation, equity curve, drawdown, and cumulative returns are all vectorised pandas/NumPy operations

---

## Tech Stack

- **Python 3.10+**
- **pandas**, **NumPy** — data manipulation and vectorised indicator/metrics calculation
- **tqdm** — progress tracking with multiprocessing support (`process_map`)
- **pyarrow / Parquet** — columnar storage for large trade result datasets
- **OANDA REST / STREAM API** — live market data and trade execution
- **logging** — structured, per-process log files

---

## Skills Demonstrated

| Area | Evidence |
|---|---|
| System design | Strategy-agnostic engine via dependency injection; clear separation of data loading, simulation, and reporting |
| Performance engineering | `float32` memory optimisation, vectorised NumPy operations, explicit garbage collection, multiprocessing |
| Fault tolerance | Checkpoint/resume system; exponential backoff with jitter; per-class exception handling with safe defaults |
| Quantitative finance | Walk-Forward Analysis, Sharpe ratio, CAGR, max drawdown, profit factor, parameter robustness analysis |
| Production readiness | Configurable logging, typed function signatures, resumable long-running jobs |

---

## Full System Overview

The complete Victory101 system includes:
- Data pipeline for 128+ instruments (Parquet, float32 optimised)
- 60+ concurrent backtesting processes via ProcessPoolExecutor
- Live trading bot with real-time Oanda API streaming
- 5-tier structured logging and observability framework
- Streamlit and Power BI dashboards for performance analysis

Available to walk through in detail during a technical conversation.

## System Output Preview
Strategy Performance Dashboard
<img width="3783" height="5175" alt="Dashboard_Demo_1" src="https://github.com/user-attachments/assets/c469a427-097c-48fc-80a3-3c8b56a80402" />
