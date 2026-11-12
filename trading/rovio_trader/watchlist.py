"""The editable watchlist. Stored in state/watchlist.json so the dashboard, the command line and the
worker all share one list. When the file exists it replaces the watchlists in config.toml.
The worker picks up a change on its next cycle; the replay is a fixed scripted run and ignores it."""
from __future__ import annotations

import json
import os
import re

from .instruments import FUTURES_SPECS

MARKETS = {"stock": ("stock", "watchlist"), "forex": ("forex", "pairs"), "crypto": ("crypto", "symbols"), "futures": ("futures", "roots")}
CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}


def normalize(market: str, symbol: str) -> str:
    """Canonical symbol for a market, or ValueError explaining what is accepted."""
    s = (symbol or "").strip().upper().replace(" ", "")
    if market == "stock":
        if not re.fullmatch(r"[A-Z]{1,5}([.\-][A-Z]{1,2})?", s):
            raise ValueError("Stock/ETF tickers are 1-5 letters, optionally a class suffix (e.g. AAPL, BRK.B)")
        return s
    if market in ("forex", "crypto"):
        s = s.replace("-", "/").replace("_", "/")
        if "/" not in s and len(s) == 6:
            s = s[:3] + "/" + s[3:]
        if market == "crypto" and "/" not in s:
            s += "/USD"
        if not re.fullmatch(r"[A-Z0-9]{2,6}/[A-Z]{3}", s):
            raise ValueError("Use BASE/QUOTE, e.g. EUR/USD or BTC/USD")
        base, quote = s.split("/")
        if market == "forex" and not (base in CCY and quote in CCY and base != quote):
            raise ValueError(f"Forex pairs must combine two of {', '.join(sorted(CCY))}")
        if market == "forex" and "USD" not in (base, quote):
            raise ValueError("Only pairs that include USD are supported (profit and loss is converted to USD through the pair itself)")
        if market == "crypto" and quote != "USD":
            raise ValueError("Only USD-quoted crypto pairs are supported")
        return s
    if market == "futures":
        if s not in FUTURES_SPECS:
            raise ValueError(f"Supported futures roots: {', '.join(FUTURES_SPECS)} (contract months are resolved automatically)")
        return s
    raise ValueError(f"Unknown market '{market}' (use stock, forex, crypto or futures)")


class Watchlist:
    def __init__(self, cfg):
        self.cfg, self.path = cfg, cfg.state_path / "watchlist.json"
        self._mtime = None

    def _from_cfg(self) -> dict:
        d = {m: list(self.cfg.markets[k][f]) for m, (k, f) in MARKETS.items()}
        d["etfs"] = list(self.cfg.markets.stock.etfs)
        return d

    def get(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                base = self._from_cfg()
                return {k: [str(x) for x in d.get(k, base[k])] for k in base}
            except (ValueError, OSError):
                pass
        return self._from_cfg()

    def save(self, d: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, self.path)

    def add(self, market: str, symbol: str, etf: bool = False) -> str:
        s = normalize(market, symbol)
        d = self.get()
        if s not in d[market]:
            if len(d[market]) >= 200:
                raise ValueError("Watchlist limit of 200 symbols per market reached")
            d[market].append(s)
        if market == "stock":
            d["etfs"] = [x for x in d["etfs"] if x != s] + ([s] if etf else [])
        self.save(d)
        return s

    def remove(self, market: str, symbol: str) -> str:
        s = normalize(market, symbol)
        d = self.get()
        d[market] = [x for x in d[market] if x != s]
        if market == "stock":
            d["etfs"] = [x for x in d["etfs"] if x != s]
        self.save(d)
        return s

    def apply(self) -> bool:
        """Copy the current lists into the config object. Returns True if anything changed."""
        d, changed = self.get(), False
        for m, (k, f) in MARKETS.items():
            if list(self.cfg.markets[k][f]) != d[m]:
                self.cfg.markets[k][f] = d[m]
                changed = True
        if list(self.cfg.markets.stock.etfs) != d["etfs"]:
            self.cfg.markets.stock.etfs = d["etfs"]
            changed = True
        return changed
