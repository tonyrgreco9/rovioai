"""The five strategies. One implementation each, used unchanged by replay, backtest, paper and live.

Bars are DataFrames indexed by bar END time (UTC) with a `start` column. A bar exists for a strategy
only once its end time has passed. Every computation is causal: a value at bar t uses bars ending at
or before t (tests recompute on truncated history to prove it). A signal carries the time of the bar
close that produced it and can only execute at a later tradable quote.

Each signal defines: entry (market order at the next tradable quote, after the gate), initial
protective stop (volatility-based), target or trailing exit, maximum holding period, expiration, and
the conditions that cancel an unfilled entry. All decisions are deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STRATEGY_MARKET
from .events import EarningsEvent
from .indicators import atr, ema, sma
from .instruments import ET, Instrument, at_et

UTC = "UTC"


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def chk(name: str, value, threshold, passed: bool) -> dict:
    f = lambda v: round(float(v), 6) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else (None if isinstance(v, float) else v)
    return {"name": name, "value": f(value), "threshold": f(threshold), "passed": bool(passed)}


def upto(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Completed bars only."""
    return df.iloc[: int(df.index.searchsorted(now, side="right"))]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a coarser UTC-aligned timeframe. A coarse bar ends at boundary + rule, so it only
    becomes visible after all of its constituents have completed."""
    g = df.groupby(df["start"].dt.floor(rule).rename("bucket"))
    out = pd.DataFrame({"start": g["start"].first(), "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index = pd.DatetimeIndex(out.index, tz=UTC) + pd.Timedelta(rule)
    return out


def asof(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Value of `series` (indexed by availability time) as known at each time in `index`."""
    pos = series.index.searchsorted(index, side="right") - 1
    vals = series.to_numpy()
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)


class Strategy:
    name = ""
    timeframe = "1d"

    def __init__(self, cfg):
        self.cfg, self.p = cfg, cfg.strategies[self.name]
        self.market = STRATEGY_MARKET[self.name]
        self._cache: dict = {}
        self._frames: dict = {}

    def applies(self, inst: Instrument) -> bool:
        return inst.market == self.market

    def _table(self, inst: Instrument, df: pd.DataFrame, extra=None):
        key = (inst.symbol, id(df), len(df))
        if self._cache.get(inst.symbol, (None,))[0] != key:
            self._cache[inst.symbol] = (key, self.build(inst, df, extra))
        return self._cache[inst.symbol][1]

    def signals(self, inst, df, since, now, extra=None) -> list[dict]:
        sigs, _ = self._table(inst, df, extra)
        return [s for s in sigs if since < pd.Timestamp(s["time"]) <= now]

    def skips(self, inst, df, since, now, extra=None) -> list[dict]:
        _, skips = self._table(inst, df, extra)
        return [s for s in skips if since < pd.Timestamp(s["time"]) <= now]

    def build(self, inst, df, extra) -> tuple[list[dict], list[dict]]:
        raise NotImplementedError

    def snapshot(self, inst, df, now) -> list[dict]:
        """Pass/fail of each condition on the latest completed bar (for the scanner)."""
        return []

    def _row(self, inst, df, now):
        """Indicator row for the latest completed bar, from a frame computed once per data version.
        Safe because every column is causal (asserted by the tests)."""
        key = (id(df), len(df))
        hit = self._frames.get(inst.symbol)
        if hit is None or hit[0] != key:
            hit = self._frames[inst.symbol] = (key, self._frame(df))
        n = int(df.index.searchsorted(now, side="right"))
        return hit[1].iloc[n - 1] if n else None

    def exit_check(self, trade: dict, inst, df, now) -> str | None:
        return None

    def entry_checks(self, sig: dict, inst, quote: dict, now, ctx) -> list[dict]:
        """Strategy-specific pre-entry checks in the instrument's own units."""
        return []

    def _base(self, inst, t, side, close, stop, why, exit_rule, inputs, **kw) -> dict:
        sid = kw.pop("id", None) or f"{self.name}:{inst.symbol}:{iso(t)}"
        return {"id": sid, "strategy": self.name, "symbol": inst.symbol, "side": side, "time": iso(t), "ref_price": float(close),
                "stop": inst.round_price(stop), "target": None, "atr": None, "expires": None, "hold_until": None, "flatten_at": None,
                "why": why, "exit_rule": exit_rule, "inputs": inputs, "meta": {}, **kw}


# ============================================================ A. stocks/ETFs: daily trend breakout
class Breakout(Strategy):
    name, timeframe = "breakout", "1d"

    def _frame(self, df):
        p = self.p
        f = pd.DataFrame(index=df.index)
        f["close"], f["volume"] = df["close"], df["volume"]
        f["prior_high"] = df["high"].rolling(p.lookback, min_periods=p.lookback).max().shift(1)
        f["prior_vol"] = df["volume"].rolling(p.volume_lookback, min_periods=p.volume_lookback).mean().shift(1)
        f["sma"] = sma(df["close"], p.sma_period)
        f["atr"] = atr(df, p.atr_period)
        f["vol_ratio"] = f["volume"] / f["prior_vol"].replace(0, np.nan)
        f["ok"] = (f["close"] > f["prior_high"]) & (f["vol_ratio"] > p.volume_mult) & (f["close"] > f["sma"]) & (f["atr"] > 0)
        return f

    def _inputs(self, r):
        p = self.p
        return [chk(f"close > prior {p.lookback}-session high", r["close"], r["prior_high"], r["close"] > r["prior_high"]),
                chk(f"volume / {p.volume_lookback}-session average", r["vol_ratio"], p.volume_mult, r["vol_ratio"] > p.volume_mult),
                chk(f"close > {p.sma_period}-day average", r["close"], r["sma"], r["close"] > r["sma"])]

    def build(self, inst, df, extra):
        p, out = self.p, []
        if len(df) < max(p.lookback, p.volume_lookback, p.sma_period, p.atr_period) + 1:
            return [], []
        f = self._frame(df)
        for t, r in f[f["ok"]].iterrows():
            stop = r["close"] - p.stop_atr_mult * r["atr"]
            if stop <= 0:
                continue
            out.append(self._base(
                inst, t, 1, r["close"], stop, atr=float(r["atr"]), expires=iso(t + pd.Timedelta(days=5)),
                why=f"Close {r['close']:.2f} above the prior {p.lookback}-session high {r['prior_high']:.2f}; volume {r['vol_ratio']:.1f}x the "
                    f"{p.volume_lookback}-session average; close above the {p.sma_period}-day average {r['sma']:.2f}.",
                exit_rule=f"Stop {stop:.2f} ({p.stop_atr_mult:g} x ATR). Trailing exit: sell at the next open after a close below the prior "
                          f"{p.exit_channel}-session low. Max hold {p.max_hold_sessions} sessions. Entry valid for the next regular session only; "
                          f"cancelled if the gap exceeds {p.max_gap_pct:g}% or the spread exceeds {p.max_spread_bps:g} bps.",
                inputs=self._inputs(r), meta={"max_hold_sessions": p.max_hold_sessions}))
        return out, []

    def snapshot(self, inst, df, now):
        d = upto(df, now)
        need = max(self.p.lookback, self.p.volume_lookback, self.p.sma_period, self.p.atr_period) + 1
        if len(d) < need:
            return [chk("history (sessions)", len(d), need, False)]
        return self._inputs(self._row(inst, df, now))

    def exit_check(self, trade, inst, df, now):
        return _daily_exit(trade, upto(df, now), lambda d: _channel_exit(d, self.p.exit_channel))

    def entry_checks(self, sig, inst, quote, now, ctx):
        return _stock_entry_checks(self.p, sig, quote)


def _channel_exit(d: pd.DataFrame, n: int, side: int = 1) -> str | None:
    if len(d) < n + 1:
        return None
    c = float(d["close"].iloc[-1])
    if side > 0:
        floor = float(d["low"].iloc[-n - 1:-1].min())
        return f"close {c:.2f} below the prior {n}-bar low {floor:.2f}" if c < floor else None
    ceil = float(d["high"].iloc[-n - 1:-1].max())
    return f"close {c:.2f} above the prior {n}-bar high {ceil:.2f}" if c > ceil else None


def _daily_exit(trade, d, rule) -> str | None:
    opened = pd.Timestamp(trade["opened"])
    held = int((d.index > opened).sum())          # completed sessions since entry, counting the entry session
    if held == 0:
        return None
    why = rule(d)
    if why:
        return "exit rule: " + why
    mx = trade["meta"].get("max_hold_sessions")
    return f"maximum holding period reached ({held} sessions)" if mx and held >= mx else None


def _stock_entry_checks(p, sig, quote):
    mid = (quote["bid"] + quote["ask"]) / 2
    gap = (quote["ask"] / sig["ref_price"] - 1) * 100
    spread = (quote["ask"] - quote["bid"]) / mid * 1e4
    return [chk("opening gap % vs signal close (abs)", abs(gap), p.max_gap_pct, abs(gap) <= p.max_gap_pct),
            chk("spread (bps)", spread, p.max_spread_bps, spread <= p.max_spread_bps)]


# ============================================================ B. stocks: post-earnings momentum
class Earnings(Strategy):
    name, timeframe = "earnings", "1d"

    def applies(self, inst):
        return inst.market == "stock" and not inst.is_etf     # ETFs have no earnings

    def evaluate(self, ev: EarningsEvent, inst, df) -> tuple[str, object, pd.Timestamp | None]:
        """-> ("signal", signal, t) | ("skip", reason, t) | ("wait", None, None) while the reaction session is incomplete."""
        p = self.p
        dates = df.index.tz_convert(ET).normalize().tz_localize(None)
        rel = pd.Timestamp(ev.release_date) if ev.release_date else None
        try:
            # First FULL regular session after publication: the release date itself for a before-open
            # release, the next session for an after-close (or unverified) release.
            pos = int(dates.searchsorted(rel, side="left" if ev.session == "bmo" else "right"))
        except (TypeError, ValueError):
            return "skip", "unparseable release date", (df.index[-1] if len(df) else None)
        if pos >= len(df):
            return "wait", None, None
        t = df.index[pos]
        bad = ev.verify()
        if bad:
            return "skip", bad, t
        if pos < max(p.volume_lookback, p.atr_period) + 1:
            return "skip", "not enough price history before the release", t
        surprise = ev.eps_actual - ev.eps_estimate
        pct = surprise / abs(ev.eps_estimate) * 100 if ev.eps_estimate else (np.inf if surprise > 0 else -np.inf)
        react, pre_close = df.iloc[pos], float(df["close"].iloc[pos - 1])
        avg_vol = float(df["volume"].iloc[pos - p.volume_lookback:pos].mean())
        c, vr = float(react["close"]), (float(react["volume"]) / avg_vol if avg_vol > 0 else 0.0)
        inputs = [chk(f"EPS surprise % ({ev.actual_basis} actual {ev.eps_actual:g} vs consensus {ev.eps_estimate:g})", pct, p.min_surprise_pct, surprise > 0 and pct > p.min_surprise_pct),
                  chk("reaction-session close > pre-release close", c, pre_close, c > pre_close),
                  chk(f"reaction volume / {p.volume_lookback}-session average", vr, p.volume_mult, vr > p.volume_mult)]
        day, when = f"{dates[pos]:%Y-%m-%d}", "before the open" if ev.session == "bmo" else "after the close"
        if not inputs[0]["passed"]:
            return "skip", f"no qualifying beat: actual {ev.eps_actual:g} vs consensus {ev.eps_estimate:g} ({ev.actual_basis})", t
        if not inputs[1]["passed"]:
            return "skip", f"reaction session {day} closed {c:.2f}, not above the pre-release close {pre_close:.2f}", t
        if not inputs[2]["passed"]:
            return "skip", f"reaction-session volume {vr:.1f}x average, needs >{p.volume_mult:g}x", t
        a = float(atr(df.iloc[:pos + 1], p.atr_period).iloc[-1])
        stop = min(float(react["low"]), c - p.min_stop_atr * a)
        if not np.isfinite(stop) or stop <= 0:
            return "skip", "could not compute a protective stop", t
        return "signal", self._base(
            inst, t, 1, c, stop, id=ev.id, atr=a, expires=iso(t + pd.Timedelta(days=5)),
            why=f"Reported {when} on {ev.release_date}: EPS {ev.eps_actual:g} vs consensus {ev.eps_estimate:g} ({ev.actual_basis}; estimate as of "
                f"{ev.estimate_asof}; surprise +{pct:.1f}%). Confirmation bar {day}: closed {c:.2f} above the pre-release close {pre_close:.2f} "
                f"on {vr:.1f}x average volume.",
            exit_rule=f"Stop {stop:.2f} (reaction-session low, at least {p.min_stop_atr:g} x ATR). Exit at the next open after a close below the "
                      f"pre-release close {pre_close:.2f} (gap filled). Max hold {p.max_hold_sessions} sessions. Entry valid for the next regular session only.",
            inputs=inputs, meta={"max_hold_sessions": p.max_hold_sessions, "pre_release_close": pre_close, "release": ev.release_date,
                                 "release_session": ev.session, "eps_actual": ev.eps_actual, "eps_estimate": ev.eps_estimate,
                                 "eps_basis": ev.actual_basis, "estimate_asof": ev.estimate_asof, "confirmation_bar": day, "source": ev.source}), t

    def build(self, inst, df, events):
        sigs, skips = [], []
        for ev in events or []:
            kind, val, t = self.evaluate(ev, inst, df)
            if kind == "signal":
                sigs.append(val)
            elif kind == "skip" and t is not None:
                skips.append({"id": ev.id, "strategy": self.name, "symbol": inst.symbol, "side": 1, "time": iso(t), "reason": val,
                              "meta": {"release": ev.release_date, "release_session": ev.session, "eps_actual": ev.eps_actual, "eps_estimate": ev.eps_estimate}})
        return sigs, skips

    def exit_check(self, trade, inst, df, now):
        ref = trade["meta"].get("pre_release_close")
        def rule(d):
            c = float(d["close"].iloc[-1])
            return f"close {c:.2f} below the pre-release close {ref:.2f} (gap filled)" if ref and c < ref else None
        return _daily_exit(trade, upto(df, now), rule)

    def entry_checks(self, sig, inst, quote, now, ctx):
        return _stock_entry_checks(self.p, sig, quote)


# ============================================================ C. futures: session opening-range breakout
class OpeningRange(Strategy):
    name, timeframe = "orb", "5m"

    def build(self, inst, df, extra):
        p, out = self.p, []
        if len(df) < p.atr_period + 2:
            return [], []
        a = atr(df, p.atr_period)
        trend = ema(df["close"], p.trend_ema)
        et_start = df["start"].dt.tz_convert(ET)
        day = et_start.dt.normalize()
        mins = et_start.dt.hour * 60 + et_start.dt.minute
        prior_close = None
        for d, idx in df.groupby(day.to_numpy()).groups.items():
            g, m = df.loc[idx], mins.loc[idx]
            rth = g[(m >= 570) & (m < 960)]
            session_prior_close, prior_close = prior_close, (float(rth["close"].iloc[-1]) if len(rth) else prior_close)
            rng = g[(m >= 570) & (m < 570 + p.range_minutes)]
            if len(rng) < p.range_minutes // p.bar_minutes:
                continue                                    # incomplete opening range: no trade today
            hi, lo = float(rng["high"].max()), float(rng["low"].min())
            anchor = rng.index[0]
            last_entry, flatten = at_et(anchor, p.last_entry), at_et(anchor, p.flatten)
            taken = 0
            for t, bar in g[(m >= 570 + p.range_minutes)].iterrows():
                if t > last_entry or taken >= p.max_trades_per_day:
                    break
                c, av = float(bar["close"]), float(a[t]) if np.isfinite(a[t]) else 0.0
                side = 1 if c > hi else -1 if c < lo else 0
                if not side or av <= 0 or (side < 0 and not p.allow_short):
                    continue
                if p.trend_filter == "ema":
                    ref, label = float(trend[t]) if np.isfinite(trend[t]) else np.nan, f"{p.trend_ema}-bar EMA"
                elif p.trend_filter == "prior_close":
                    ref, label = (session_prior_close if session_prior_close is not None else np.nan), "prior session close"
                else:
                    ref, label = c - side, "none"
                trend_ok = np.isfinite(ref) and (c - ref) * side > 0
                inputs = [chk("close outside the opening range " + ("(above high)" if side > 0 else "(below low)"), c, hi if side > 0 else lo, True),
                          chk(f"trend filter: close vs {label}", c, ref, trend_ok)]
                if not trend_ok:
                    continue
                dist = p.stop_atr_mult * av
                stop, target = c - side * dist, c + side * p.target_r * dist
                taken += 1
                out.append(self._base(
                    inst, t, side, c, stop, target=inst.round_price(target), atr=av, flatten_at=iso(flatten),
                    expires=iso(min(t + pd.Timedelta(minutes=p.entry_valid_min), last_entry)),
                    why=f"{'Long' if side > 0 else 'Short'}: 5-minute bar closed {c:.2f} {'above' if side > 0 else 'below'} the first-{p.range_minutes}-minute "
                        f"range {lo:.2f}-{hi:.2f}; trend filter ({label} {ref:.2f}) agrees.",
                    exit_rule=f"Stop {stop:.2f} ({p.stop_atr_mult:g} x ATR), target {target:.2f} ({p.target_r:g}R). Flatten at {p.flatten} ET. "
                              f"No entries after {p.last_entry} ET; an unfilled entry is cancelled after {p.entry_valid_min:g} minutes.",
                    inputs=inputs, meta={"range_high": hi, "range_low": lo, "contract": inst.symbol, "expiry": inst.expiry}))
        return out, []

    def exit_check(self, trade, inst, df, now):
        return None     # stop, target and the session flatten are handled generically by the engine

    def entry_checks(self, sig, inst, quote, now, ctx):
        p = self.p
        spread_ticks = (quote["ask"] - quote["bid"]) / inst.tick_size
        px = quote["ask"] if sig["side"] > 0 else quote["bid"]
        slip = (px - sig["ref_price"]) * sig["side"] / sig["atr"]
        days = (pd.Timestamp(inst.expiry, tz=UTC) - now.normalize()).days
        excl = self.cfg.markets.futures.expiry_exclusion_days
        return [chk("spread (ticks)", spread_ticks, p.max_spread_ticks, spread_ticks <= p.max_spread_ticks),
                chk("adverse move since the signal (ATR)", slip, p.max_slippage_atr, slip <= p.max_slippage_atr),
                chk(f"days to expiry of {inst.symbol} (exclusion window)", days, excl, days > excl),
                chk(f"before last-entry time {p.last_entry} ET", f"{now.tz_convert(ET):%H:%M}", p.last_entry, now <= at_et(now, p.last_entry))]


# ============================================================ D. spot forex: trend pullback
class FxPullback(Strategy):
    name, timeframe = "fx_pullback", "1h"

    def _frame(self, df):
        p = self.p
        h4 = resample(df, "4h")
        fast, slow = ema(h4["close"], p.fast_ema), ema(h4["close"], p.slow_ema)
        f = pd.DataFrame(index=df.index)
        f["open"], f["close"], f["low"], f["high"] = df["open"], df["close"], df["low"], df["high"]
        f["fast4"], f["slow4"] = asof(fast, df.index), asof(slow, df.index)
        f["ema"] = ema(df["close"], p.pullback_ema)
        f["atr"] = atr(df, p.atr_period)
        f["dir"] = np.sign(f["fast4"] - f["slow4"]).fillna(0)
        touch_long = (f["low"] <= f["ema"]) | (f["low"].shift(1) <= f["ema"].shift(1))
        touch_short = (f["high"] >= f["ema"]) | (f["high"].shift(1) >= f["ema"].shift(1))
        f["long"] = (f["dir"] > 0) & touch_long & (f["close"] > f["ema"]) & (f["close"] > f["open"])
        f["short"] = (f["dir"] < 0) & touch_short & (f["close"] < f["ema"]) & (f["close"] < f["open"])
        return f

    def build(self, inst, df, extra):
        p, out = self.p, []
        if len(df) < 4 * p.slow_ema + 4:
            return [], []
        f = self._frame(df)
        last = {1: None, -1: None}
        for t, r in f[(f["long"] | f["short"]) & (f["atr"] > 0)].iterrows():
            side = 1 if r["long"] else -1
            if side < 0 and not p.allow_short:
                continue
            if last[side] is not None and t - last[side] < pd.Timedelta(hours=6):
                continue                                    # one signal per pullback
            last[side] = t
            dist = p.stop_atr_mult * r["atr"]
            stop, target = r["close"] - side * dist, r["close"] + side * p.target_r * dist
            d = inst.pip and int(round(-np.log10(inst.pip))) + 1 or 5
            out.append(self._base(
                inst, t, side, r["close"], stop, target=inst.round_price(target), atr=float(r["atr"]),
                expires=iso(t + pd.Timedelta(minutes=p.entry_valid_min)), hold_until=iso(t + pd.Timedelta(hours=p.max_hold_hours)),
                why=f"{'Long' if side > 0 else 'Short'}: 4-hour EMA{p.fast_ema} {r['fast4']:.{d}f} {'above' if side > 0 else 'below'} EMA{p.slow_ema} "
                    f"{r['slow4']:.{d}f}; hourly price pulled back to its EMA{p.pullback_ema} ({r['ema']:.{d}f}) and closed back {'above' if side > 0 else 'below'} it at {r['close']:.{d}f}.",
                exit_rule=f"Stop {stop:.{d}f} ({p.stop_atr_mult:g} x hourly ATR), target {target:.{d}f} ({p.target_r:g}R), max hold {p.max_hold_hours:g}h. "
                          f"Unfilled entry cancelled after {p.entry_valid_min:g} minutes, if the spread exceeds {p.max_spread_pips:g} pips, or inside a calendar blackout.",
                inputs=[chk(f"4h EMA{p.fast_ema} vs EMA{p.slow_ema} (trend {'up' if side > 0 else 'down'})", r["fast4"], r["slow4"], True),
                        chk(f"pullback touched hourly EMA{p.pullback_ema}", r["low"] if side > 0 else r["high"], r["ema"], True),
                        chk("close back in the trend direction", r["close"], r["ema"], True)]))
        return out, []

    def snapshot(self, inst, df, now):
        d = upto(df, now)
        if len(d) < 4 * self.p.slow_ema + 4:
            return [chk("history (hourly bars)", len(d), 4 * self.p.slow_ema + 4, False)]
        r = self._row(inst, df, now)
        return [chk(f"4h EMA{self.p.fast_ema} vs EMA{self.p.slow_ema}", r["fast4"], r["slow4"], r["dir"] != 0),
                chk("pullback and close back in trend direction", r["close"], r["ema"], bool(r["long"] or r["short"]))]

    def entry_checks(self, sig, inst, quote, now, ctx):
        p = self.p
        spread = (quote["ask"] - quote["bid"]) / inst.pip
        px = quote["ask"] if sig["side"] > 0 else quote["bid"]
        slip = (px - sig["ref_price"]) * sig["side"] / sig["atr"]
        out = [chk("spread (pips)", spread, p.max_spread_pips, spread <= p.max_spread_pips),
               chk("adverse move since the signal (ATR)", slip, p.max_slippage_atr, slip <= p.max_slippage_atr)]
        cal = ctx.get("calendar")
        why = cal.block_reason([inst.base_ccy, inst.quote_ccy], now, p.blackout_before_min, p.blackout_after_min,
                               self.cfg.markets.forex.calendar_max_age_hours) if cal is not None else "economic calendar unavailable: forex entries are paused"
        out.append(chk("economic calendar: " + (why or "fresh, no blackout"), None, None, why is None))
        return out


# ============================================================ E. Bitcoin: volatility breakout
class BtcBreakout(Strategy):
    name, timeframe = "btc_breakout", "4h"

    def _frame(self, df):
        p = self.p
        daily = resample(df, "1D")
        full = daily
        dsma = sma(full["close"], p.daily_trend_sma)
        f = pd.DataFrame(index=df.index)
        f["close"] = df["close"]
        f["prior_high"] = df["high"].rolling(p.lookback, min_periods=p.lookback).max().shift(1)
        f["atr"] = atr(df, p.atr_period)
        f["dclose"], f["dsma"] = asof(full["close"], df.index), asof(dsma, df.index)
        f["ok"] = (f["close"] > f["prior_high"]) & (f["dclose"] > f["dsma"]) & (f["atr"] > 0)
        return f

    def _inputs(self, r):
        p = self.p
        return [chk(f"4h close > prior {p.lookback}-bar high", r["close"], r["prior_high"], r["close"] > r["prior_high"]),
                chk(f"daily trend: last UTC daily close > {p.daily_trend_sma}-day average", r["dclose"], r["dsma"], r["dclose"] > r["dsma"])]

    def build(self, inst, df, extra):
        p, out = self.p, []
        if len(df) < 6 * p.daily_trend_sma + 6:
            return [], []
        f = self._frame(df)
        for t, r in f[f["ok"]].iterrows():
            stop = r["close"] - p.stop_atr_mult * r["atr"]
            if stop <= 0:
                continue
            out.append(self._base(
                inst, t, 1, r["close"], stop, atr=float(r["atr"]), expires=iso(t + pd.Timedelta(minutes=p.entry_valid_min)),
                hold_until=iso(t + pd.Timedelta(hours=4 * p.max_hold_bars)),
                why=f"4-hour close {r['close']:,.2f} above the prior {p.lookback}-bar high {r['prior_high']:,.2f}; daily trend positive "
                    f"(last UTC daily close {r['dclose']:,.2f} above its {p.daily_trend_sma}-day average {r['dsma']:,.2f}).",
                exit_rule=f"Stop {stop:,.2f} ({p.stop_atr_mult:g} x ATR). Trailing exit after a 4h close below the prior {p.exit_channel}-bar low. "
                          f"Max hold {p.max_hold_bars} bars. Unfilled entry cancelled after {p.entry_valid_min:g} minutes or if the spread exceeds {p.max_spread_bps:g} bps.",
                inputs=self._inputs(r)))
        return out, []

    def snapshot(self, inst, df, now):
        d = upto(df, now)
        if len(d) < 6 * self.p.daily_trend_sma + 6:
            return [chk("history (4h bars)", len(d), 6 * self.p.daily_trend_sma + 6, False)]
        return self._inputs(self._row(inst, df, now))

    def exit_check(self, trade, inst, df, now):
        d = upto(df, now)
        if not len(d) or d.index[-1] <= pd.Timestamp(trade["opened"]):
            return None
        why = _channel_exit(d, self.p.exit_channel)
        return "exit rule: " + why if why else None

    def entry_checks(self, sig, inst, quote, now, ctx):
        p = self.p
        mid = (quote["bid"] + quote["ask"]) / 2
        spread = (quote["ask"] - quote["bid"]) / mid * 1e4
        slip = (quote["ask"] - sig["ref_price"]) / sig["atr"]
        return [chk("spread (bps)", spread, p.max_spread_bps, spread <= p.max_spread_bps),
                chk("adverse move since the signal (ATR)", slip, p.max_slippage_atr, slip <= p.max_slippage_atr)]


ALL = {c.name: c for c in (Breakout, Earnings, OpeningRange, FxPullback, BtcBreakout)}


def enabled(cfg, only=None) -> list[Strategy]:
    return [cls(cfg) for name, cls in ALL.items() if cfg.strategies[name].enabled and (only is None or name in only)]
