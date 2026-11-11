"""Technical indicators. All functions take/return pandas Series aligned to the input index."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder(-d.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window -> RSI 100 (flat series stays NaN -> 50).
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    return out.where(~((loss == 0) & (gain == 0)), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = _wilder(true_range(df), n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / tr
    minus_di = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx.fillna(0.0), n)


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d, min_periods=d).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0.0) * df["volume"]).cumsum()


def donchian(df: pd.DataFrame, n: int = 20):
    """Prior-n-bar channel (excludes the current bar, so a breakout is detectable)."""
    return df["high"].rolling(n, min_periods=n).max().shift(1), df["low"].rolling(n, min_periods=n).min().shift(1)


def roc(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the standard indicator set added."""
    o = df.copy()
    c = o["close"]
    o["sma20"], o["sma50"], o["sma200"] = sma(c, 20), sma(c, 50), sma(c, 200)
    o["ema12"], o["ema26"] = ema(c, 12), ema(c, 26)
    o["rsi"] = rsi(c)
    o["macd"], o["macd_signal"], o["macd_hist"] = macd(c)
    o["bb_mid"], o["bb_up"], o["bb_lo"] = bollinger(c)
    o["bb_pctb"] = (c - o["bb_lo"]) / (o["bb_up"] - o["bb_lo"]).replace(0, np.nan)
    o["atr"] = atr(o)
    o["adx"] = adx(o)
    o["stoch_k"], o["stoch_d"] = stochastic(o)
    o["obv"] = obv(o)
    o["don_hi20"], o["don_lo20"] = donchian(o, 20)
    o["don_hi55"], o["don_lo55"] = donchian(o, 55)
    o["roc63"] = roc(c, 63)
    return o
