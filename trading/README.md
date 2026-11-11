# Rovio Trader

A local trading workstation: charts and indicators, a transparent buy/sell signal, strategy
backtests, risk-managed order proposals, and a broker layer. Covers stocks, ETFs, mutual funds,
forex pairs and crypto. Python standard library + pandas/numpy; no build step.

> **Read this first.** Signals here are rule-based heuristics, not predictions. No software can
> promise trading profits; most active strategies underperform simply holding an index fund after
> costs and taxes, and the built-in backtester will often show you exactly that. Start on the
> paper broker. Nothing in this project is financial advice.

## Quick start

```bash
cd trading
pip install -r requirements.txt
python -m rovio_trader --demo serve      # synthetic prices, no network: a safe tour
python -m rovio_trader serve             # real (delayed) market data, paper broker
```

Open http://127.0.0.1:8787. Other commands: `scan`, `analyze SYMBOL`, `backtest SYMBOL`,
`propose`, `portfolio`, and `run` (scheduled scans). Run tests with
`python -m unittest discover -s tests`.

Symbols use Yahoo Finance notation: `AAPL`, `SPY`, `FXAIX` (mutual fund), `BTC-USD`,
`EURUSD=X` (typing `EUR/USD` or `BTC/USD` also works).

## What it does

| Piece | Details |
|---|---|
| Charts | Candles, SMA 50/200, Bollinger bands, volume, RSI, MACD; crosshair, zoom, daily/weekly/hourly |
| Signal | Score from -100 to +100: a weighted vote of trend, 50/200 cross, MACD, 63-bar momentum, RSI, Bollinger, volume flow. Every vote is shown with its reason |
| Levels | ATR stop and target, 20-bar support and resistance |
| Backtest | 5 strategies vs buy & hold. Fills at the *next* bar's open, 10 bps cost per side, out-of-sample split, trade markers on the chart |
| Risk | Position sized so an ATR stop loses 1% of equity; 20% max per position; max positions; daily loss stop; kill switch. Long-only, no margin |
| Workflow | Scan -> sized proposals -> you approve or reject -> broker. Every order passes the risk checks |

## Brokers

Set `ROVIO_BROKER` in a local `.env` (copy `.env.example`; `.env` is gitignored and must never be committed).

- **`paper`** (default): built-in simulator with $100k of pretend cash.
- **`manual`**: for accounts with no trading API, **including Fidelity**. Approving a proposal
  creates an order ticket; you place it yourself in Fidelity, then record the fill price so the
  tracked portfolio stays accurate.
- **`alpaca`**: sends orders through the Alpaca API (US stocks/ETFs and crypto). Uses Alpaca's
  paper endpoint unless `ALPACA_LIVE=true`. Keys are read from your local `.env` only.

### About Fidelity

Fidelity does not offer a trading API for retail accounts, so this project cannot place orders in
a Fidelity account. The only way to automate it would be scripting the website with your login,
which breaks Fidelity's customer agreement, can get the account restricted, and means handing
your credentials to a script. That is deliberately not implemented. Use the `manual` broker with
Fidelity, or open an account at a broker that has an official API for the automated portion.

### Automated trading

```bash
python -m rovio_trader run --every 900            # scan every 15 min, queue proposals for approval
python -m rovio_trader run --every 900 --auto     # also execute them (paper / Alpaca paper)
python -m rovio_trader run --auto --confirm-live  # required when ALPACA_LIVE=true
```

Run `--auto` on paper for weeks before considering real money, and keep the size small if you do.

## Limits you should know about

- Market data is Yahoo Finance's unofficial public endpoint: delayed, unguaranteed, and it can
  break or rate-limit without notice. For live trading use your broker's or a paid feed
  (implement `Provider._fetch` in `rovio_trader/data.py`).
- The Yahoo and Alpaca network paths are covered by parser/request tests against recorded
  shapes, but were not exercised against the live services during development.
- Mutual funds trade once a day at NAV and often penalise round trips inside 30-90 days.
- Forex is analysis-only in practice: spreads, leverage and financing are not modelled.
- Stops are suggested levels; the platform does not place stop orders at the broker.
- Taxes (short-term gains, wash sales) are not modelled.
- The dashboard has no login. It binds to 127.0.0.1; do not expose it to a network.
