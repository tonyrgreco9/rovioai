# Rovio Trader

A personal multi-asset trading bot: it scans four markets, explains why each opportunity qualifies,
sizes the position, sends orders automatically, manages exits, and journals what happened.

> **Profitability is a hypothesis here, not a feature.** Every strategy starts as *research only*.
> The bot is built to decide there is no trade, and the backtester is built to tell you when a
> strategy does not survive costs. Nothing in this repository is financial advice.

## What runs today

| Market | Broker | Replay tested | Paper adapter | Paper connection verified | Live adapter | Still needed |
|---|---|---|---|---|---|---|
| US stocks / ETFs | Alpaca | yes | yes | **no** (never connected) | same adapter, never exercised | Alpaca paper keys; earnings CSV for the earnings strategy |
| Bitcoin BTC/USD | Alpaca | yes | yes | **no** | same adapter, never exercised | Alpaca paper keys with crypto enabled |
| Spot forex | OANDA v20 | yes | yes | **no** | same adapter, never exercised | OANDA Practice token + USD account; economic-calendar CSV |
| Futures MES / MNQ | Interactive Brokers | yes | **not built** | no | not built | IBKR adapter; TWS/IB Gateway paper; CME data |
| Fidelity | - | n/a | **not available** | no | not available | No official retail order-entry API found |

`python -m rovio_trader status` prints this table from the actual state on your machine. A connection
is marked verified only after a real broker response; a simulated fill never counts.

The Alpaca and OANDA adapters were written from the brokers' published API references and tested
against local stand-ins. **They have never talked to the real services.** Your first
`check alpaca` / `check oanda` is the real verification; expect to fix small mismatches.

## Setup

Python 3.11+.

```bash
cd trading
pip install -r requirements.txt
python -m rovio_trader init                 # creates state/ with paper.db and live.db
cp config.example.toml config.toml          # optional: edit limits and watchlists
cp .env.example .env                        # optional: add PAPER credentials on your machine only
```

Windows PowerShell: use `copy` instead of `cp`.

## Commands

| Goal | Command |
|---|---|
| Replay all four markets, no credentials | `python -m rovio_trader replay` |
| Dashboard for the replay | `python -m rovio_trader serve --mode replay` then open http://127.0.0.1:8787 |
| Run the tests | `python -m unittest discover -s tests -t .` |
| Sizing math (stock, MES, USD/JPY, Bitcoin) | `python -m rovio_trader sizing-demo` |
| Backtest one strategy | `python -m rovio_trader backtest breakout --data csv:mydata [--tune] [--save]` |
| Pipeline check without data | `python -m rovio_trader backtest all` (synthetic prices; never evidence) |
| Connect Alpaca paper | add keys to `.env`, then `python -m rovio_trader check alpaca` |
| Connect OANDA Practice | add token + account id to `.env`, then `python -m rovio_trader check oanda` |
| IBKR / Fidelity status | `python -m rovio_trader check ibkr` / `check fidelity` |
| Start the paper worker | `python -m rovio_trader worker` |
| Dashboard for paper | `python -m rovio_trader serve --mode paper` |
| Controls | `pause on|off`, `kill on|off`, `flatten [trade ids]`, `resume` (also buttons on the dashboard) |

The worker is a separate long-running process: it keeps scanning, reconciling and managing exits
with the browser closed. Keep it alive with your OS tools (`nohup python -m rovio_trader worker &`,
a systemd unit, or Windows Task Scheduler). It does not run inside a Claude session. If it stops,
protective stop orders already at the broker stay there, but rule-based exits wait until it restarts.

## The five strategies (initial research settings, all configurable)

| | Market | Signal | Stop | Exit | Expires |
|---|---|---|---|---|---|
| `breakout` | stocks/ETFs, daily | close > prior 20-session high, volume > 1.5x average, close > 50-day average | 2 x ATR | close below prior 10-session low; 30 sessions | next regular session; gap > 3% or spread > 20 bps cancels |
| `earnings` | stocks (no ETFs), daily | verified EPS beat, first full session closes above pre-release close on > 1.5x volume | reaction-day low (>= 1 ATR) | close below pre-release close; 40 sessions | next regular session |
| `orb` | MES/MNQ, 5-min | bar closes outside the first-30-minute range, trend filter agrees; long or short | 3 x ATR | 2R target; flatten 15:45 ET | 5 minutes; none after 14:30 ET |
| `fx_pullback` | EUR/USD, GBP/USD, USD/JPY | 4h EMA20 vs EMA50 trend, 1h pullback to EMA20 and close back; long or short | 2 x ATR | 2R target; 72 hours | 15 minutes; spread > 2.5 pips or calendar blackout cancels |
| `btc_breakout` | BTC/USD, 4h UTC | close > prior 20-bar high, last UTC daily close above its 20-day average; long only | 2.5 x ATR | close below prior 10-bar low; 30 bars | 30 minutes; spread > 30 bps cancels |

Signals use completed bars only and execute at a later tradable quote. Decisions are deterministic;
nothing AI-generated can place an order.

## Data you must supply

- **Prices (paper/live):** each instrument's own broker (Alpaca IEX feed by default: its volume is one
  venue's share of the tape, which makes the stock volume filter noisier; SIP is a paid subscription).
- **Prices (backtests):** CSV files, `<dir>/<SYMBOL>_<tf>.csv` with `start,open,high,low,close,volume`
  (`EUR/USD` -> `EUR-USD_1h.csv`; timeframes `1d`, `5m`, `1h`, `4h`). Use split/dividend-adjusted
  stock data. Futures backtests use one continuous series per root (`MES_5m.csv`), research only.
- **Earnings history:** `examples/earnings.example.csv` shows the columns. Point-in-time consensus
  (the estimate as it stood *before* the release, with a timestamp and GAAP/adjusted basis) is a paid
  product at the vendors we know of. Until you configure a file the strategy produces no signals and
  the dashboard says so. Rows missing a release time, estimate timestamp or matching basis are
  skipped with the reason shown.
- **Economic calendar (forex):** `examples/calendar.example.csv`. Refresh it daily; if the file is
  older than 24 hours, forex entries pause.

## Risk and execution, in one page

- One USD risk view across brokers; cash and margin checked per broker.
- Size = planned loss at the stop (plus estimated costs) within the per-trade and total-open-risk
  budgets, then capped by instrument, correlated group (SPY+MES, QQQ+MNQ), market, broker, leverage,
  order size, currency exposure, cash/margin and liquidity. Rounded down; if the smallest legal size
  is too big the trade is skipped with the reason.
- Risk is reserved and the order intent saved in one database transaction *before* the broker is
  contacted. After a timeout or restart the bot looks the order up by its client id before retrying.
- Protective stops always equal the filled quantity. If a stop cannot be placed: entries halt, the
  unfilled remainder is cancelled, the configured emergency policy runs, and you get an alert.
- Daily-loss and drawdown halts include unrealized P&L, persist across restarts, stop new entries and
  keep exits managed. Daily reset: 17:00 New York, every day.
- Pause entries / kill switch / flatten are separate. Flatten touches only positions the bot opened.
  A manual holding in a watched symbol blocks automation in that symbol (the broker nets them).
- **A stop is not a guarantee.** Gaps and slippage can lose more than the planned amount; the replay
  includes a trade that lost about 7x its plan.

## Going live (deliberately hard)

Live needs, all at once: your own limits with `risk.set_by_user = true`; separate `*_LIVE_*`
credentials; a saved backtest on real data that clears the evidence bar for that exact strategy
version and settings (`backtest <strategy> --data csv:<dir> --save`); and
`activate-live --broker --market --strategy`, which shows the account and limits and makes you type
a confirmation. Then `worker --live` trades the activated combinations automatically with no
per-order approval. Changing strategy or risk settings invalidates the activation. No live order was
sent during development.

## Fidelity

Checked 2026-11-12 against public sources: no officially supported way was found to place orders in
a retail Fidelity account from your own software. Fidelity Access is consented, read-only account
data sharing for approved aggregators. Order-capable connectivity (Integration Xchange / Wealthscape,
FIX) belongs to Fidelity Institutional's custody-and-clearing clients. This software has never
connected to Fidelity, does not know your account type, never asks for your Fidelity password, and
does not automate the website. `check fidelity` lists what would have to be true to change this.

## Engineering decisions worth knowing

- Kept the existing stack (standard-library HTTP server, pandas, SQLite) instead of adding FastAPI.
- One engine and one set of strategy/sizing functions serve replay, backtest, paper and live.
- Replay, paper and live use separate journals (`state/replay.db`, `paper.db`, `live.db`).
- Simulator assumptions: market orders fill at the next bar's open plus half-spread and slippage;
  stops fill at the stop, or at the open if the bar gaps through; when a bar touches stop and target
  the stop wins; an order fills at most 1% of a bar's volume.
- Profit targets and rule exits are managed by the worker (cancel stop, then one reduce-only market
  order), not as resting broker orders. Only the protective stop rests at the broker.
- Alpaca crypto protection uses stop-limit orders (no plain stop there); a fast move can skip the limit.
- Stocks trade in whole shares, long only. Backtests are not tax-aware.
- The earlier composite-score dashboard and manual order tickets were removed: they were a second
  order path outside these risk controls.
