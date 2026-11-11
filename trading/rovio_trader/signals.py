"""Composite signal: a transparent weighted vote of standard indicators, scored -100..+100.

The same vectorised score drives the live signal and the 'composite' backtest strategy, so
what you backtest is what you get. It is a rules-based heuristic, not a prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import compute_all

WEIGHTS = {"trend": 20, "cross": 20, "macd": 15, "momentum": 15, "rsi": 10, "bollinger": 10, "volume": 10}
BUY, STRONG = 20, 50
MIN_BARS = 60


def votes(ind: pd.DataFrame) -> pd.DataFrame:
    """Per-bar component votes in [-1, 1] plus the weighted 'score'. NaN inputs vote 0."""
    c = ind["close"]
    chop = np.where(ind["adx"] < 20, 0.5, 1.0)  # trend votes count half in a trendless market
    v = pd.DataFrame(index=ind.index)
    v["trend"] = np.sign(c - ind["sma200"]) * chop
    v["cross"] = np.sign(ind["sma50"] - ind["sma200"]) * chop
    v["macd"] = np.sign(ind["macd_hist"])
    v["momentum"] = np.sign(ind["roc63"])
    r = ind["rsi"]
    v["rsi"] = ((r - 50) / 20).clip(-1, 1).where((r > 25) & (r < 75), 0.0)  # extremes: no chase
    v["bollinger"] = np.where(ind["bb_pctb"] < 0, 1.0, np.where(ind["bb_pctb"] > 1, -1.0, 0.0))
    v.loc[ind["bb_pctb"].isna(), "bollinger"] = np.nan
    obv_slope = ind["obv"] - ind["obv"].shift(20)
    v["volume"] = np.sign(obv_slope).where(ind["volume"].rolling(20).sum() > 0, 0.0)
    v = v.fillna(0.0)
    v["score"] = sum(v[k] * w for k, w in WEIGHTS.items())
    return v


def action_for(score: float) -> str:
    if score >= STRONG:
        return "STRONG BUY"
    if score >= BUY:
        return "BUY"
    if score <= -STRONG:
        return "STRONG SELL"
    if score <= -BUY:
        return "SELL"
    return "HOLD"


def _f(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def analyze(df: pd.DataFrame, atr_stop_mult: float = 2.0) -> dict:
    """Signal for the latest bar, with the reasoning behind every component."""
    if len(df) < MIN_BARS:
        return {"action": "NO DATA", "score": 0, "components": [], "warnings": [f"Only {len(df)} bars; need {MIN_BARS}+"]}
    ind = compute_all(df)
    v = votes(ind)
    x, vv = ind.iloc[-1], v.iloc[-1]
    c = float(x["close"])
    warnings = []
    if len(df) < 200:
        warnings.append("Under 200 bars of history: long-term trend components are inactive.")

    def cmp(a, b, up, down):
        return "n/a (not enough history)" if not np.isfinite(b) else (up if a > b else down)

    reasons = {
        "trend": cmp(c, x["sma200"], "Price above 200-bar average (uptrend)", "Price below 200-bar average (downtrend)"),
        "cross": cmp(x["sma50"], x["sma200"], "50-bar average above 200-bar (golden-cross regime)", "50-bar average below 200-bar (death-cross regime)"),
        "macd": cmp(x["macd_hist"], 0.0, "MACD above its signal line", "MACD below its signal line"),
        "momentum": cmp(x["roc63"], 0.0, f"Up {x['roc63']:.1%} over 63 bars", f"Down {abs(x['roc63']):.1%} over 63 bars"),
        "rsi": f"RSI {x['rsi']:.0f}" + (" (overbought; not chasing)" if x["rsi"] >= 75 else " (oversold; not shorting)" if x["rsi"] <= 25 else ""),
        "bollinger": "Below lower Bollinger band (stretched down)" if x["bb_pctb"] < 0 else "Above upper Bollinger band (stretched up)" if x["bb_pctb"] > 1 else "Inside Bollinger bands",
        "volume": "No volume data" if df["volume"].tail(20).sum() == 0 else ("Volume flow rising (accumulation)" if vv["volume"] > 0 else "Volume flow falling (distribution)" if vv["volume"] < 0 else "Volume flow flat"),
    }
    if np.isfinite(x["adx"]) and x["adx"] < 20:
        warnings.append(f"ADX {x['adx']:.0f}: weak trend, choppy conditions; trend votes are halved.")
    comps = [{"name": k, "weight": w, "vote": _f(vv[k], 2), "points": _f(vv[k] * w, 1), "reason": reasons[k]} for k, w in WEIGHTS.items()]
    a = float(x["atr"]) if np.isfinite(x["atr"]) else 0.0
    prev = float(ind["close"].iloc[-2])
    return {
        "action": action_for(float(vv["score"])),
        "score": _f(vv["score"], 1),
        "price": _f(c, 6),
        "change_pct": _f((c / prev - 1) * 100, 2),
        "as_of": ind.index[-1].isoformat(),
        "components": comps,
        "warnings": warnings,
        "levels": {
            "atr": _f(a, 6),
            "atr_pct": _f(a / c * 100, 2),
            "stop": _f(c - atr_stop_mult * a, 6),
            "target": _f(c + 1.5 * atr_stop_mult * a, 6),
            "support": _f(x["don_lo20"], 6),
            "resistance": _f(x["don_hi20"], 6),
        },
        "indicators": {k: _f(x[k], 4) for k in ("rsi", "adx", "macd_hist", "sma50", "sma200", "stoch_k", "bb_pctb")},
    }
