"""Price feed for the manual (order-ticket) mode: Yahoo Finance's public chart endpoint.

It is unofficial and unsupported. It can change, rate-limit or stop without notice, and prices can be
delayed (up to about 15 minutes, depending on the exchange). It reports the last trade only, never a
bid or ask. It is used because the manual mode has no broker connection to take data from; it never
receives account details or credentials.
"""
from __future__ import annotations

import time
import urllib.parse

import pandas as pd

from ..data import daily_frame, frame
from ..instruments import ET
from .base import BrokerError, BrokerTimeout
from .http import Http

URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
RANGE = {"5m": "5d", "1h": "3mo", "1d": "2y"}
STEP = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1)}
LABEL = "Yahoo Finance chart feed (unofficial): last trade price, can be delayed up to ~15 min, no bid/ask"


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-").replace("/", "-")       # BRK.B -> BRK-B


def parse(payload: dict, tf: str) -> dict | None:
    """-> {"meta": {...}, "bars": DataFrame} or None when Yahoo does not know the symbol."""
    chart = (payload or {}).get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        return None
    res = chart["result"][0]
    meta = res.get("meta") or {}
    ts = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    raw = pd.DataFrame({k: pd.to_numeric(pd.Series(q.get(k) or [None] * len(ts), dtype="object"), errors="coerce") for k in ("open", "high", "low", "close", "volume")})
    raw["t"] = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
    raw = raw.dropna(subset=["open", "high", "low", "close"])
    raw["volume"] = raw["volume"].fillna(0.0)
    if tf == "1d":
        # One row per session. The row for a session still in progress carries the latest trade time, so key on the exchange date.
        raw["d"] = raw["t"].dt.tz_convert(meta.get("exchangeTimezoneName") or ET).dt.date
        raw = raw.drop_duplicates("d", keep="last").sort_values("d")
        bars = daily_frame(raw["d"], raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"])
    else:
        raw = raw.drop_duplicates("t", keep="last").sort_values("t")
        start = pd.DatetimeIndex(raw["t"])
        bars = frame(start, start + STEP[tf], raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"])
    return {"meta": meta, "bars": bars}


class YahooFeed:
    def __init__(self, http=None, ttl: float = 12.0, daily_ttl: float = 300.0):
        self.http = http or Http({"User-Agent": "Mozilla/5.0"})
        self.ttl, self.daily_ttl, self._cache = ttl, daily_ttl, {}
        self.verified = False                      # True after one real, parseable response in this process

    def chart(self, symbol: str, tf: str) -> dict | None:
        key = (symbol, tf)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < (self.daily_ttl if tf == "1d" else self.ttl):
            return hit[1]
        try:
            payload = self.http.call("GET", URL + urllib.parse.quote(yahoo_symbol(symbol)), params={"interval": tf, "range": RANGE[tf]}, ok_404=True)
        except BrokerError as e:
            raise BrokerTimeout(f"price feed refused the request ({e})") from None     # feed trouble is never a trading decision
        except BrokerTimeout as e:
            raise BrokerTimeout(str(e).replace("broker", "the price feed")) from None
        out = parse(payload, tf) if payload else None
        if out is not None:
            self.verified = True
        self._cache[key] = (time.time(), out)
        return out
