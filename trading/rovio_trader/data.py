"""Market data providers.

YahooProvider covers stocks, ETFs, mutual funds, FX (EURUSD=X) and crypto (BTC-USD) through
Yahoo Finance's public chart endpoint. It is unofficial, delayed, and can change or rate-limit
without notice; swap in a paid feed by implementing `history()`.

SyntheticProvider generates deterministic fake prices for demos and tests. It must never be
used to make real trading decisions.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import zlib

import numpy as np
import pandas as pd

RANGE_DAYS = {"1mo": 30, "3mo": 91, "6mo": 182, "1y": 365, "2y": 730, "5y": 1826, "10y": 3652}
INTERVALS = {"1h": 3600, "1d": 86400, "1wk": 604800}


class DataError(Exception):
    pass


def normalize_symbol(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "")
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.\-=/]{0,14}", s):
        raise DataError(f"Invalid symbol: {raw!r}")
    if "/" in s:
        base, quote = s.split("/", 1)
        fiat = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
        return f"{base}{quote}=X" if base in fiat and quote in fiat else f"{base}-{quote}"
    return s


def asset_class(symbol: str) -> str:
    """Heuristic asset class from the symbol format: forex | crypto | mutual_fund | equity."""
    if symbol.endswith("=X"):
        return "forex"
    if re.fullmatch(r"[A-Z0-9]+-(USD|USDT|EUR)", symbol):
        return "crypto"
    if re.fullmatch(r"[A-Z]{4}X", symbol):
        return "mutual_fund"
    return "equity"


def _check(range_: str, interval: str) -> None:
    if range_ not in RANGE_DAYS:
        raise DataError(f"Unsupported range {range_!r}; use one of {sorted(RANGE_DAYS)}")
    if interval not in INTERVALS:
        raise DataError(f"Unsupported interval {interval!r}; use one of {sorted(INTERVALS)}")
    if interval == "1h" and RANGE_DAYS[range_] > 730:
        raise DataError("Hourly data is limited to the last 2 years")


class Provider:
    name = "base"
    is_synthetic = False

    def __init__(self, ttl: float = 60.0):
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self.ttl = ttl

    def history(self, symbol: str, range_: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """OHLCV frame indexed by UTC timestamp, oldest first."""
        symbol = normalize_symbol(symbol)
        _check(range_, interval)
        key = (symbol, range_, interval)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        df = self._fetch(symbol, range_, interval)
        if df.empty:
            raise DataError(f"No data returned for {symbol}")
        self._cache[key] = (time.time(), df)
        return df

    def last_price(self, symbol: str) -> float:
        return float(self.history(symbol, "1mo", "1d")["close"].iloc[-1])

    def _fetch(self, symbol: str, range_: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError


class YahooProvider(Provider):
    name = "yahoo"
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={r}&interval={i}&includeAdjustedClose=true"

    def _fetch(self, symbol, range_, interval):
        url = self.URL.format(sym=urllib.parse.quote(symbol), r=range_, i=interval)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.load(resp)
        except Exception as e:  # network, HTTP, JSON
            raise DataError(f"Market data request failed for {symbol}: {e}") from e
        return parse_yahoo_chart(payload, symbol)


def parse_yahoo_chart(payload: dict, symbol: str = "") -> pd.DataFrame:
    chart = payload.get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        raise DataError(f"No data for {symbol}: {(chart.get('error') or {}).get('description', 'empty result')}")
    res = chart["result"][0]
    ts = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame(
        {k: pd.to_numeric(pd.Series(q.get(k, [None] * len(ts)), dtype="object"), errors="coerce")
         for k in ("open", "high", "low", "close", "volume")}
    )
    df.index = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
    df.index.name = "time"
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if adj and len(adj) == len(df):
        # Adjust for splits/dividends so indicators and backtests see total-return prices.
        f = (pd.to_numeric(pd.Series(adj, dtype="object"), errors="coerce").to_numpy() / df["close"].to_numpy())
        for k in ("open", "high", "low", "close"):
            df[k] = df[k] * f
    df = df.dropna(subset=["close"])
    for k in ("open", "high", "low"):
        df[k] = df[k].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df[~df.index.duplicated(keep="last")].sort_index()


class SyntheticProvider(Provider):
    """Deterministic regime-switching random walk, seeded per symbol. DEMO/TEST ONLY."""
    name = "synthetic"
    is_synthetic = True

    def __init__(self, end: str | None = None, ttl: float = 3600.0):
        super().__init__(ttl)
        self.end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").floor("D")

    def _fetch(self, symbol, range_, interval):
        cls = asset_class(symbol)
        step = INTERVALS[interval]
        start = self.end - pd.Timedelta(days=RANGE_DAYS[range_])
        idx = pd.date_range(start, self.end, freq=pd.Timedelta(seconds=step), tz="UTC")
        if cls != "crypto":
            idx = idx[idx.dayofweek < 5]
            if interval == "1h":
                idx = idx[(idx.hour >= 14) & (idx.hour < 21)]
        # Generate on a fixed epoch grid so overlapping ranges agree with each other.
        rng = np.random.default_rng(zlib.crc32(f"{symbol}|{interval}".encode()))
        n_total = 12000
        ann_vol = {"crypto": 0.65, "forex": 0.08, "mutual_fund": 0.14}.get(cls, 0.28)
        per_year = {"1h": 252 * 7, "1d": 252, "1wk": 52}[interval] * (365 / 252 if cls == "crypto" and interval != "1wk" else 1)
        sd = ann_vol / np.sqrt(per_year)
        regime = np.repeat(rng.choice([-0.35, 0.1, 0.45], size=n_total // 120 + 1, p=[0.25, 0.35, 0.4]), 120)[:n_total]
        rets = regime / per_year + sd * rng.standard_normal(n_total)
        base = {"crypto": 30000.0, "forex": 1.1, "mutual_fund": 150.0}.get(cls, 120.0)
        path = np.cumsum(rets)
        close_all = base * np.exp(path - path[-1])  # series ends near `base`
        noise = np.abs(rng.standard_normal((n_total, 3))) * sd
        vol_all = rng.lognormal(15, 0.4, n_total)
        n = len(idx)
        if n > n_total:
            raise DataError("Synthetic range too long")
        c = close_all[-n:]
        o = np.concatenate([[c[0]], c[:-1]]) * np.exp((noise[-n:, 0] - sd * 0.8) * 0.3)
        h = np.maximum(o, c) * np.exp(noise[-n:, 1] * 0.6)
        l = np.minimum(o, c) * np.exp(-noise[-n:, 2] * 0.6)
        v = vol_all[-n:]
        if cls == "mutual_fund":  # one NAV per day
            o = h = l = c
            v = np.zeros(n)
        if cls == "forex":
            v = np.zeros(n)
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx.rename("time"))
