"""Bar data sources. Bars are DataFrames indexed by bar END (UTC) with columns
start, open, high, low, close, volume. A bar is usable only once its end has passed.

  FrameData    in-memory frames (replay fixtures, backtests, tests)
  CsvData      your own history: <dir>/<SYMBOL>_<tf>.csv  (EUR/USD -> EUR-USD_1h.csv)
  BrokerData   live/paper: bars come from each instrument's own broker adapter
  synthetic()  random-walk bars for plumbing checks ONLY; never evidence of anything
"""
from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import ET

TF = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
COLS = ["start", "open", "high", "low", "close", "volume"]


class DataError(Exception):
    pass


def frame(starts, ends, o, h, l, c, v) -> pd.DataFrame:
    df = pd.DataFrame({"start": pd.DatetimeIndex(starts), "open": np.asarray(o, float), "high": np.asarray(h, float), "low": np.asarray(l, float),
                       "close": np.asarray(c, float), "volume": np.asarray(v, float)})
    df.index = pd.DatetimeIndex(ends, name="end")
    return df


def session_times(dates) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """US regular session open/close (UTC) for calendar dates."""
    d = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    op = (d + pd.Timedelta(hours=9, minutes=30)).tz_localize(ET).tz_convert("UTC")
    cl = (d + pd.Timedelta(hours=16)).tz_localize(ET).tz_convert("UTC")
    return op, cl


def daily_frame(dates, o, h, l, c, v) -> pd.DataFrame:
    op, cl = session_times(dates)
    return frame(op, cl, o, h, l, c, v)


class FrameData:
    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame], label: str):
        self.frames, self.label = frames, label

    def bars(self, inst, tf: str) -> pd.DataFrame | None:
        return self.frames.get((inst.symbol, tf))


def _fname(symbol: str, tf: str) -> str:
    return f"{symbol.replace('/', '-')}_{tf}.csv"


class CsvData(FrameData):
    """CSV columns: start,open,high,low,close,volume[,end]. `start` is UTC; for daily stock bars a plain
    date is accepted and mapped to the 09:30-16:00 ET session."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        super().__init__({}, f"CSV files in {self.dir.name}/ (you are responsible for adjustments and accuracy)")

    def bars(self, inst, tf):
        key = (inst.symbol, tf)
        if key not in self.frames:
            p = self.dir / _fname(inst.symbol, tf)
            if not p.is_file():
                return None
            raw = pd.read_csv(p)
            miss = [c for c in COLS if c not in raw.columns]
            if miss:
                raise DataError(f"{p.name} is missing columns {miss}")
            if tf == "1d" and inst.market == "stock":
                op, cl = session_times(pd.to_datetime(raw["start"]).dt.date)
            else:
                op = pd.DatetimeIndex(pd.to_datetime(raw["start"], utc=True))
                cl = pd.DatetimeIndex(pd.to_datetime(raw["end"], utc=True)) if "end" in raw.columns else op + TF[tf]
            self.frames[key] = frame(op, cl, raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"]).sort_index()
        return self.frames[key]


def write_csv(directory: Path, symbol: str, tf: str, df: pd.DataFrame):
    directory.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    for c in ("start", "end"):
        out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out[COLS + ["end"]].to_csv(directory / _fname(symbol, tf), index=False)


class BrokerData:
    """Paper/live: each instrument's bars come from its own broker, cached briefly. Incomplete bars are dropped."""
    LIMIT = {"5m": 400, "1h": 500, "4h": 400, "1d": 300}

    def __init__(self, brokers: dict, ttl: float = 20.0):
        self.brokers, self.ttl, self._cache = brokers, ttl, {}
        self.label = "; ".join(f"{k}: {b.data_label}" for k, b in brokers.items())

    def bars(self, inst, tf):
        key = (inst.symbol, tf)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        b = self.brokers[inst.broker]
        df = b.bars(inst, tf, self.LIMIT[tf])
        if df is not None and len(df):
            df = df[df.index <= b.now()]
        self._cache[key] = (time.time(), df)
        return df


def synthetic(symbol: str, market: str, tf: str, end: str, periods: int) -> pd.DataFrame:
    """Deterministic regime-switching random walk. For checking that the pipeline runs; NOT market data."""
    rng = np.random.default_rng(zlib.crc32(f"{symbol}|{tf}".encode()))
    end_ts = pd.Timestamp(end, tz="UTC")
    if tf == "1d" and market == "stock":
        days = pd.bdate_range(end=end_ts.tz_localize(None).normalize(), periods=periods)
        op, cl = session_times(days)
    elif market == "futures":
        days = pd.bdate_range(end=end_ts.tz_localize(None).normalize(), periods=max(periods // 78, 1))
        o0, _ = session_times(days)
        op = pd.DatetimeIndex([o + TF[tf] * i for o in o0 for i in range(78)])
        cl = op + TF[tf]
    else:
        op = pd.date_range(end=end_ts.floor(TF[tf]) - TF[tf], periods=periods, freq=TF[tf], tz="UTC")
        if market == "forex":
            op = op[op.dayofweek < 5]
        cl = op + TF[tf]
    n = len(op)
    per_year = {"1d": 252, "4h": 6 * 365, "1h": 24 * 260, "5m": 78 * 252}[tf]
    vol = {"stock": 0.3, "futures": 0.18, "forex": 0.08, "crypto": 0.6}[market]
    sd = vol / np.sqrt(per_year)
    regime = np.repeat(rng.choice([-0.3, 0.05, 0.4], size=n // 150 + 1, p=[0.25, 0.35, 0.4]), 150)[:n]
    path = np.cumsum(regime / per_year + sd * rng.standard_normal(n))
    base = {"stock": 120.0, "futures": 6000.0, "forex": 1.1, "crypto": 60000.0}[market] * (130 if symbol.endswith("JPY") else 1)
    c = base * np.exp(path - path[-1])
    o = np.concatenate([[c[0]], c[:-1]]) * np.exp(sd * 0.2 * rng.standard_normal(n))
    wig = np.abs(rng.standard_normal((n, 2))) * sd * 0.7
    h, l = np.maximum(o, c) * np.exp(wig[:, 0]), np.minimum(o, c) * np.exp(-wig[:, 1])
    v = rng.lognormal(14, 0.5, n) if market in ("stock", "futures", "crypto") else np.zeros(n)
    return frame(op, cl, o, h, l, c, v)
