"""Instrument specifications and the money math that depends on them.

P&L in quote currency = qty * multiplier * (exit - entry) * side. `usd_per_quote` converts to USD.
Specs here are FALLBACK reference values for replay and backtest; in paper/live each adapter's
`instrument()` returns what the broker itself reports, and that is what gets used for sizing.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

ET = "America/New_York"


@dataclass
class Instrument:
    symbol: str                 # AAPL | BTC/USD | EUR/USD | MESZ5 (a dated contract, never a continuous series)
    market: str                 # stock | futures | forex | crypto
    broker: str
    multiplier: float = 1.0     # quote-currency value of a 1.0 price move, per unit
    tick_size: float = 0.01
    qty_step: float = 1.0
    min_qty: float = 1.0
    quote_ccy: str = "USD"
    base_ccy: str = ""
    shortable: bool = False
    margin_rate: float = 1.0    # stocks/crypto: 1.0 (cash). forex: broker margin rate
    margin_per_unit: float = 0.0  # futures: initial margin per contract (USD)
    root: str = ""              # futures root (MES)
    expiry: str = ""            # futures last trade date YYYY-MM-DD
    exchange: str = ""
    is_etf: bool = False
    pip: float = 0.0            # forex pip size (0.0001, or 0.01 for JPY pairs)

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier

    def round_qty(self, qty: float) -> float:
        """Round DOWN to the broker increment."""
        steps = math.floor(qty / self.qty_step + 1e-9)
        return round(steps * self.qty_step, 10)

    def round_price(self, px: float) -> float:
        return round(round(px / self.tick_size) * self.tick_size, 10)

    def to_dict(self) -> dict:
        return asdict(self)


FUTURES_SPECS = {   # CME micro equity index futures (reference; confirm with the broker's contract details)
    "MES": {"multiplier": 5.0, "tick_size": 0.25, "exchange": "CME", "margin_per_unit": 2500.0, "group_symbol": "MES"},
    "MNQ": {"multiplier": 2.0, "tick_size": 0.25, "exchange": "CME", "margin_per_unit": 3800.0, "group_symbol": "MNQ"},
}
MONTH_CODES = "FGHJKMNQUVXZ"


def stock(symbol: str, broker: str, is_etf: bool = False) -> Instrument:
    return Instrument(symbol, "stock", broker, is_etf=is_etf)


def crypto(symbol: str, broker: str) -> Instrument:
    base, quote = symbol.split("/")
    return Instrument(symbol, "crypto", broker, tick_size=0.01, qty_step=0.0001, min_qty=0.0001, base_ccy=base, quote_ccy=quote)


def forex(symbol: str, broker: str) -> Instrument:
    base, quote = symbol.split("/")
    pip = 0.01 if quote == "JPY" else 0.0001
    return Instrument(symbol, "forex", broker, tick_size=pip / 10, qty_step=1.0, min_qty=1.0, base_ccy=base, quote_ccy=quote,
                      shortable=True, margin_rate=0.05, pip=pip)


def future(root: str, expiry: str, broker: str) -> Instrument:
    spec = FUTURES_SPECS[root]
    d = pd.Timestamp(expiry)
    sym = f"{root}{MONTH_CODES[d.month - 1]}{d.year % 10}"
    return Instrument(sym, "futures", broker, multiplier=spec["multiplier"], tick_size=spec["tick_size"], shortable=True,
                      margin_per_unit=spec["margin_per_unit"], root=root, expiry=expiry, exchange=spec["exchange"])


def third_friday(year: int, month: int) -> str:
    d = pd.Timestamp(year=year, month=month, day=1)
    return (d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)).strftime("%Y-%m-%d")


def quarterly_contracts(root: str, broker: str, around: pd.Timestamp, n: int = 3) -> list[Instrument]:
    """Reference quarterly cycle (Mar/Jun/Sep/Dec, third Friday) for replay/backtest."""
    out, y, m = [], around.year, around.month
    while len(out) < n:
        if m % 3 == 0 and pd.Timestamp(third_friday(y, m), tz="UTC") + pd.Timedelta(hours=14) >= around:
            out.append(future(root, third_friday(y, m), broker))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def active_contract(contracts: list[Instrument], now: pd.Timestamp, exclusion_days: int) -> Instrument | None:
    """The nearest contract that is still outside the expiry exclusion window."""
    ok = [c for c in contracts if (pd.Timestamp(c.expiry, tz="UTC") - now.normalize()).days > exclusion_days]
    return min(ok, key=lambda c: c.expiry) if ok else None


def group_symbol(inst: Instrument) -> str:
    """Symbol used for concentration and correlated-exposure limits (futures use their root)."""
    return inst.root or inst.symbol


def usd_per_quote(ccy: str, rates: dict[str, float]) -> float:
    """USD value of one unit of `ccy`. `rates` maps pair symbols to mid prices (e.g. 'USD/JPY': 150.0)."""
    if ccy == "USD":
        return 1.0
    if f"{ccy}/USD" in rates:
        return rates[f"{ccy}/USD"]
    if f"USD/{ccy}" in rates and rates[f"USD/{ccy}"] > 0:
        return 1.0 / rates[f"USD/{ccy}"]
    raise KeyError(f"No conversion rate for {ccy} to USD")


def et_time(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert(ET)


def at_et(ts: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    """The given ET wall-clock time on ts's ET calendar date, in UTC."""
    h, m = map(int, hhmm.split(":"))
    return et_time(ts).normalize().replace(hour=h, minute=m).tz_convert("UTC")
