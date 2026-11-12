"""REPLAY FIXTURES. Every price, volume, earnings figure and calendar event in this file is INVENTED
to exercise the trading pipeline. Tickers RPLA..RPLX and MANL are fictional. Futures, forex and
Bitcoin symbols are real instrument names but the prices are scripted, not market history.
Nothing here says anything about real markets or about whether a strategy makes money."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import instruments as I
from .data import FrameData, daily_frame, frame
from .events import EarningsEvent, EarningsSource, EconCalendar

LABEL = "SIMULATED replay fixture (scripted prices, not market data)"
U = lambda s: pd.Timestamp(s, tz="UTC")


def _sessions():
    days = pd.bdate_range("2025-08-18", "2025-12-12")
    return days[~days.isin(pd.to_datetime(["2025-09-01", "2025-11-27"]))]


class _Stock:
    def __init__(self, days, base=100.0, amp=0.6):
        self.days, self.n, self.amp = days, len(days), amp
        self.level, self.vol, self.gaps = np.full(self.n, base), np.full(self.n, 1_000_000.0), {}

    def _i(self, d):
        return int(self.days.get_loc(pd.Timestamp(d)))

    def shift(self, d, pct, vol=1.0, gap=False):
        i = self._i(d)
        self.level[i:] *= 1 + pct
        self.vol[i] *= vol
        if gap:
            self.gaps[i] = True
        return self

    def drift(self, d0, d1, pct):
        for i in range(self._i(d0), self._i(d1) + 1):
            self.level[i:] *= 1 + pct
        return self

    def build(self):
        c = self.level + self.amp * np.sin(0.7 * np.arange(self.n))
        o = np.concatenate([[c[0]], c[:-1]])
        for i in self.gaps:                      # the whole move happens at the open
            o[i] = c[i] - self.amp * 0.2
        h, l = np.maximum(o, c) + 0.3 * self.amp, np.minimum(o, c) - 0.3 * self.amp
        return daily_frame(self.days, o, h, l, c, self.vol)


def _stocks():
    d = _sessions()
    s = lambda: _Stock(d)
    out = {
        # full lifecycle: breakout Nov 10 -> entry Nov 11 -> trend -> trailing-channel exit
        "RPLA": s().shift("2025-11-10", 0.035, 2.5).drift("2025-11-11", "2025-11-24", 0.009).drift("2025-11-25", "2025-11-28", -0.027),
        # skipped: breakout Nov 12, then a 6% opening gap on Nov 13
        "RPLB": s().shift("2025-11-12", 0.035, 2.5).shift("2025-11-13", 0.06, gap=True),
        # earnings only (an earlier bump keeps the 20-session high above the reaction close), then an ordinary stop-out
        "RPLE": s().shift("2025-10-27", 0.07).shift("2025-11-03", -0.065).shift("2025-11-18", 0.04, 3.0).shift("2025-11-24", -0.05),
        "RPLF": s(),                              # earnings row with an unverified release time
        # both strategies flag the same stock on the same bar; later an ordinary stop-out
        "RPLJ": s().shift("2025-11-20", 0.04, 3.0).shift("2025-12-01", -0.05),
        # entry whose broker reply is lost (timeout after acceptance); later gaps 12% through its stop
        "RPLC": s().shift("2025-12-02", 0.035, 2.5).shift("2025-12-09", -0.12, gap=True),
        "RPLD": s().shift("2025-12-03", 0.035, 2.5),          # partial fill, remainder cancelled
        "RPLG": s().shift("2025-12-08", 0.035, 2.5),          # breakout signal arriving as the loss limit trips
        "RPLH": s().shift("2025-12-08", 0.04, 3.0),           # earnings signal arriving at the same moment
        "RPLI": s().shift("2025-12-04", 0.035, 2.5),          # manual holding in a watched symbol
        "RPLX": s(),                              # an ETF (excluded from the earnings strategy)
        "MANL": s(),                              # manual holding, not on the watchlist
    }
    return {k: v.build() for k, v in out.items()}


def _hourly(start, end, base, moves: dict, pip):
    """Weekday hourly bars. `moves` maps an hour (UTC) to a price change for that bar."""
    op = pd.date_range(U(start), U(end), freq="1h")
    op = op[op.dayofweek < 5]
    delta = np.array([moves.get(t, 0.0) for t in op])
    c = base + np.cumsum(delta)
    o = np.concatenate([[base], c[:-1]])
    return frame(op, op + pd.Timedelta(hours=1), o, np.maximum(o, c) + pip, np.minimum(o, c) - pip, c, np.zeros(len(op)))


def _fx():
    def path(sign, step, t0, pull_at, resume_at, n_after):
        m = {}
        for t in pd.date_range(U(t0), U(pull_at) - pd.Timedelta(hours=1), freq="1h"):
            m[t] = sign * step
        for k in range(3):
            m[U(pull_at) + pd.Timedelta(hours=k)] = -sign * step * 4
        m[U(resume_at)] = sign * step * 5
        for k in range(1, n_after + 1):
            m[U(resume_at) + pd.Timedelta(hours=k)] = sign * step * 5 / 3
        return m
    return {
        # long: trend up from Dec 1, pullback Dec 2 10:00-12:00, resume 13:00 -> signal at 14:00 -> target
        "USD/JPY": _hourly("2025-11-10", "2025-12-05 23:00", 150.0, path(1, 0.03, "2025-12-01", "2025-12-02 10:00", "2025-12-02 13:00", 12), 0.01),
        # short: same shape down, signal at Dec 3 12:00, inside the blackout around a scripted EUR event
        "EUR/USD": _hourly("2025-11-10", "2025-12-05 23:00", 1.1, path(-1, 0.0003, "2025-12-01", "2025-12-03 08:00", "2025-12-03 11:00", 6), 0.0001),
        "GBP/USD": _hourly("2025-11-10", "2025-12-05 23:00", 1.3, {}, 0.0001),
    }


def _btc():
    op = pd.date_range(U("2025-10-20"), U("2025-12-10 20:00"), freq="4h")
    delta = np.array([100.0 if i % 2 == 0 else -100.0 for i in range(len(op))])
    idx = {t: i for i, t in enumerate(op)}
    def put(t0, vals):
        for k, v in enumerate(vals):
            delta[idx[U(t0)] + k] = v
    put("2025-12-05 00:00", [200] * 6)             # Friday: daily close turns the daily trend positive
    put("2025-12-06 00:00", [600, 300, 300, 300, 300])   # Saturday 04:00 UTC: breakout bar closes; equities are shut
    i0 = idx[U("2025-12-06 20:00")]
    for k in range(12):
        delta[i0 + k] = 50.0 if k % 2 == 0 else -50.0
    put("2025-12-08 20:00", [-700])                # close below the prior 10-bar low -> trailing exit
    c = 89950.0 + np.cumsum(delta)
    o = np.concatenate([[89950.0], c[:-1]])
    return frame(op, op + pd.Timedelta(hours=4), o, np.maximum(o, c) + 10, np.minimum(o, c) - 10, c, np.full(len(op), 500.0))


def _fut_day(date, base, script: dict | None = None, tick=0.25):
    """78 five-minute RTH bars. `script` maps a bar number to a price change; otherwise a +/-1 point wiggle."""
    op0 = (pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)).tz_localize(I.ET).tz_convert("UTC")
    op = pd.DatetimeIndex([op0 + pd.Timedelta(minutes=5 * i) for i in range(78)])
    level, c = base, []
    for i in range(78):
        if script and i in script:
            level += script[i]
        c.append(level + (1.0 if i % 2 == 0 else -1.0))
    c = np.array(c)
    o = np.concatenate([[base], c[:-1]])
    return frame(op, op + pd.Timedelta(minutes=5), o, np.maximum(o, c) + 2 * tick, np.minimum(o, c) - 2 * tick, c, np.full(78, 1000.0))


def _futures():
    up = {i: 1.5 for i in range(6, 20)}            # Dec 11: breaks above the opening range and runs to the target
    dn = {i: -1.5 for i in range(6, 9)}            # Dec 12: breaks below, then goes nowhere until the session cutoff
    cat = lambda *f: pd.concat(f)
    return {
        "MESZ5": cat(_fut_day("2025-12-10", 6850.0), _fut_day("2025-12-11", 6850.0, up)),
        "MESH6": cat(_fut_day("2025-12-11", 6900.0), _fut_day("2025-12-12", 6900.0, dn)),
        "MNQZ5": cat(_fut_day("2025-12-10", 25500.0), _fut_day("2025-12-11", 25500.0)),
        "MNQH6": cat(_fut_day("2025-12-11", 25700.0), _fut_day("2025-12-12", 25700.0)),
    }


EARNINGS = [
    EarningsEvent("RPLE", "2025-11-17", "amc", 1.32, 1.20, "adjusted", "adjusted", "2025-11-14", "replay fixture"),
    EarningsEvent("RPLF", "2025-11-18", "", 0.80, 0.70, "adjusted", "adjusted", "2025-11-14", "replay fixture"),      # release time unknown
    EarningsEvent("RPLJ", "2025-11-19", "amc", 2.10, 1.95, "gaap", "gaap", "2025-11-18", "replay fixture"),
    EarningsEvent("RPLH", "2025-12-08", "bmo", 0.64, 0.55, "adjusted", "adjusted", "2025-12-05", "replay fixture"),
    EarningsEvent("RPLX", "2025-11-18", "amc", 1.00, 0.90, "adjusted", "adjusted", "2025-11-14", "replay fixture"),   # ETF: never evaluated
]
CALENDAR = [(U("2025-12-03 12:30"), "EUR", "ECB rate decision (scripted fixture event)"),
            (U("2025-12-04 13:30"), "USD", "US jobs report (scripted fixture event)")]
START, END = U("2025-11-10 14:30"), U("2025-12-12 21:00")
WATCHLIST = ["RPLA", "RPLB", "RPLC", "RPLD", "RPLE", "RPLF", "RPLG", "RPLH", "RPLI", "RPLJ", "RPLX"]
OVERRIDES = {   # replay settings: labelled example limits sized for the scripted scenario
    "portfolio": {"sim_starting_cash": {"alpaca": 150000.0, "oanda": 50000.0, "ibkr": 50000.0}},
    "markets": {"stock": {"watchlist": WATCHLIST, "etfs": ["RPLX"]}},
    "risk": {"max_instrument_pct": 20.0, "daily_loss_limit_pct": 1.0, "max_open_risk_pct": 3.0, "max_positions": 8},
}


def build(cfg):
    """-> (instruments, exec bars by symbol, FrameData for strategies, EarningsSource, EconCalendar)"""
    stocks, fx, btc, fut = _stocks(), _fx(), _btc(), _futures()
    sb, fb, ob, cb = cfg.markets.stock.broker, cfg.markets.futures.broker, cfg.markets.forex.broker, cfg.markets.crypto.broker
    insts = [I.stock(s, sb, is_etf=s in cfg.markets.stock.etfs) for s in stocks]
    insts += [I.forex(s, ob) for s in fx] + [I.crypto("BTC/USD", cb)]
    expiry = {"Z5": "2025-12-19", "H6": "2026-03-20"}
    insts += [I.future(s[:3], expiry[s[3:]], fb) for s in fut]
    exec_bars = {**stocks, **fx, "BTC/USD": btc, **fut}
    frames = {(s, "1d"): df for s, df in stocks.items()}
    frames.update({(s, "1h"): df for s, df in fx.items()})
    frames[("BTC/USD", "4h")] = btc
    frames.update({(s, "5m"): df for s, df in fut.items()})
    earnings = EarningsSource(EARNINGS, "configured", "REPLAY FIXTURE: 5 invented earnings events")
    calendar = EconCalendar(CALENDAR, asof=None, status="configured", detail="REPLAY FIXTURE: 2 invented events")
    calendar.sim_always_fresh = True
    return insts, exec_bars, FrameData(frames, LABEL), earnings, calendar
