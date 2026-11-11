"""Long/flat backtester. Signals are computed on a bar's close and filled at the NEXT bar's
open, with commission + slippage charged on every position change. No look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import compute_all
from .signals import BUY, votes


def _latch(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """1 after an entry bar until an exit bar."""
    e, x = entry.fillna(False).to_numpy(bool), exit_.fillna(False).to_numpy(bool)
    out, pos = np.zeros(len(e)), 0.0
    for i in range(len(e)):
        if pos and x[i]:
            pos = 0.0
        elif not pos and e[i] and not x[i]:
            pos = 1.0
        out[i] = pos
    return pd.Series(out, index=entry.index)


def _composite(ind):
    s = votes(ind)["score"]
    return _latch(s >= BUY, s < 0)


STRATEGIES = {
    "composite": ("Composite score (buy >= +20, exit < 0)", _composite),
    "sma_cross": ("50/200 moving-average crossover", lambda i: (i["sma50"] > i["sma200"]).astype(float)),
    "macd": ("MACD above signal, price above 200-bar average", lambda i: ((i["macd_hist"] > 0) & (i["close"] > i["sma200"])).astype(float)),
    "rsi_reversion": ("RSI dip-buy in an uptrend (enter RSI<30, exit RSI>55)", lambda i: _latch((i["rsi"] < 30) & (i["close"] > i["sma200"]), i["rsi"] > 55)),
    "donchian": ("Donchian breakout (enter 55-bar high, exit 20-bar low)", lambda i: _latch(i["close"] > i["don_hi55"], i["close"] < i["don_lo20"])),
    "buy_hold": ("Buy and hold", lambda i: pd.Series(1.0, index=i.index)),
}


def _metrics(rets: pd.Series, per_year: float) -> dict:
    if len(rets) < 2:
        return {}
    eq = (1 + rets).cumprod()
    total = float(eq.iloc[-1] - 1)
    years = len(rets) / per_year
    sd = float(rets.std(ddof=0))
    down = float(np.sqrt((rets.clip(upper=0) ** 2).mean()))
    dd = float((eq / eq.cummax() - 1).min())
    r = lambda x, n=4: round(float(x), n)
    return {
        "total_return": r(total),
        "cagr": r((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else None,
        "volatility": r(sd * np.sqrt(per_year)),
        "sharpe": r(rets.mean() / sd * np.sqrt(per_year), 2) if sd > 0 else None,
        "sortino": r(rets.mean() / down * np.sqrt(per_year), 2) if down > 0 else None,
        "max_drawdown": r(dd),
    }


def run(df: pd.DataFrame, strategy: str = "composite", cost_bps: float = 5.0, slippage_bps: float = 5.0, oos_fraction: float = 0.3) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    if len(df) < 100:
        raise ValueError("Need at least 100 bars to backtest")
    ind = compute_all(df)
    sig = STRATEGIES[strategy][1](ind).fillna(0.0)
    pos = sig.shift(1).fillna(0.0)          # position held from this bar's open
    prev_pos = pos.shift(1).fillna(0.0)     # position carried in overnight
    o, c = df["open"], df["close"]
    pc = c.shift(1)
    gap = (o / pc - 1).fillna(0.0)
    intraday = c / o - 1
    turnover = (pos - prev_pos).abs()
    rate = (cost_bps + slippage_bps) / 1e4
    rets = (1 + prev_pos * gap) * (1 + pos * intraday) * (1 - turnover * rate) - 1
    bh = c.pct_change().fillna(0.0)

    span_years = max((df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    per_year = len(df) / span_years

    trades, entry = [], None
    for i in np.flatnonzero(turnover.to_numpy() > 0):
        px = float(o.iloc[i])
        if pos.iloc[i] > 0:
            entry = (df.index[i], px * (1 + rate))
        elif entry:
            xp = px * (1 - rate)
            trades.append({"entry_time": entry[0].isoformat(), "entry_price": round(entry[1], 6), "exit_time": df.index[i].isoformat(), "exit_price": round(xp, 6), "return": round(xp / entry[1] - 1, 4)})
            entry = None
    if entry:
        trades.append({"entry_time": entry[0].isoformat(), "entry_price": round(entry[1], 6), "exit_time": None, "exit_price": None, "return": round(float(c.iloc[-1]) / entry[1] - 1, 4), "open": True})
    closed = [t["return"] for t in trades if t["exit_time"]]
    wins, losses = [t for t in closed if t > 0], [t for t in closed if t <= 0]

    split = int(len(rets) * (1 - oos_fraction))
    stats = _metrics(rets, per_year)
    stats.update({
        "trades": len(closed),
        "win_rate": round(len(wins) / len(closed), 3) if closed else None,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "avg_trade": round(float(np.mean(closed)), 4) if closed else None,
        "exposure": round(float(pos.mean()), 3),
    })
    return {
        "strategy": strategy,
        "description": STRATEGIES[strategy][0],
        "bars": len(df),
        "start": df.index[0].isoformat(),
        "end": df.index[-1].isoformat(),
        "costs_bps_per_side": cost_bps + slippage_bps,
        "stats": stats,
        "buy_hold": _metrics(bh, per_year),
        "in_sample": _metrics(rets.iloc[:split], per_year),
        "out_of_sample": _metrics(rets.iloc[split:], per_year),
        "oos_start": df.index[split].isoformat(),
        "trades": trades,
        "equity": {
            "t": [int(t.timestamp()) for t in df.index],
            "strategy": [round(float(x), 5) for x in (1 + rets).cumprod()],
            "buy_hold": [round(float(x), 5) for x in (1 + bh).cumprod()],
        },
    }
