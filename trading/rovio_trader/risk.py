"""Position sizing and pre-trade risk checks. Every order passes through `check` before a broker sees it."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .data import asset_class


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0    # % of equity lost if the ATR stop is hit
    max_position_pct: float = 20.0     # max % of equity in one symbol
    max_open_positions: int = 10
    max_daily_loss_pct: float = 3.0    # new buys are blocked after losing this much in a day
    atr_stop_mult: float = 2.0


def round_qty(symbol: str, qty: float) -> float:
    cls = asset_class(symbol)
    if cls == "crypto":
        return math.floor(qty * 1e6) / 1e6
    if cls in ("mutual_fund", "forex"):
        return math.floor(qty * 1e3) / 1e3
    return float(math.floor(qty))


def position_size(cfg: RiskConfig, symbol: str, equity: float, cash: float, price: float, atr: float | None) -> float:
    """Units to buy so that a stop `atr_stop_mult` ATRs away risks `risk_per_trade_pct` of equity,
    capped by the per-position limit and available cash."""
    if price <= 0 or equity <= 0:
        return 0.0
    cap_dollars = min(equity * cfg.max_position_pct / 100 * 0.98, cash * 0.99)  # headroom for slippage
    qty = cap_dollars / price
    if atr and atr > 0:
        qty = min(qty, equity * cfg.risk_per_trade_pct / 100 / (cfg.atr_stop_mult * atr))
    return max(round_qty(symbol, qty), 0.0)


def check(cfg: RiskConfig, *, side: str, symbol: str, qty: float, price: float, equity: float, cash: float,
          positions: dict[str, float], kill_switch: bool, day_start_equity: float | None) -> list[str]:
    """Return a list of violations (empty = order allowed). `positions` maps symbol -> quantity held."""
    errs = []
    if kill_switch:
        errs.append("Kill switch is ON: all trading is halted.")
    if qty <= 0 or price <= 0:
        errs.append("Quantity and price must be positive.")
        return errs
    held = positions.get(symbol, 0.0)
    if side == "sell":
        if qty > held + 1e-9:
            errs.append(f"Cannot sell {qty:g} {symbol}: only {held:g} held (short selling is disabled).")
        return errs
    if side != "buy":
        return errs + [f"Unknown side {side!r}"]
    cost = qty * price
    if cost > cash + 1e-6:
        errs.append(f"Insufficient cash: order needs ${cost:,.2f}, have ${cash:,.2f} (no margin).")
    if equity > 0 and (held * price + cost) / equity * 100 > cfg.max_position_pct + 1e-6:
        errs.append(f"Position would exceed {cfg.max_position_pct:g}% of equity in {symbol}.")
    if held <= 0 and sum(1 for q in positions.values() if q > 0) >= cfg.max_open_positions:
        errs.append(f"Already at the maximum of {cfg.max_open_positions} open positions.")
    if day_start_equity and equity < day_start_equity * (1 - cfg.max_daily_loss_pct / 100):
        errs.append(f"Daily loss limit of {cfg.max_daily_loss_pct:g}% reached: no new buys today.")
    return errs
