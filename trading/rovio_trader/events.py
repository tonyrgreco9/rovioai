"""Timestamped event inputs: earnings releases (stocks) and the economic calendar (forex).

There is no built-in vendor integration for either. Point-in-time EPS consensus history is a paid
product at the providers we know of, so both are supplied as CSV files you maintain or export
(columns in README). Anything that cannot be verified is skipped with the reason shown, never guessed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class EarningsEvent:
    symbol: str
    release_date: str            # YYYY-MM-DD, exchange calendar date of publication
    session: str                 # "bmo" (before the open) | "amc" (after the close) | anything else = unverified
    eps_actual: float | None
    eps_estimate: float | None
    actual_basis: str            # "gaap" | "adjusted" | ""
    estimate_basis: str
    estimate_asof: str           # when the consensus figure was captured (date or timestamp); "" = unknown
    source: str = ""

    @property
    def id(self) -> str:
        return f"earnings:{self.symbol}:{self.release_date}"

    def verify(self) -> str | None:
        """Why this event cannot be used, or None if it is verifiable."""
        if self.session not in ("bmo", "amc"):
            return f"release time not verified (session={self.session or 'unknown'!r}; need 'bmo' or 'amc')"
        if self.eps_actual is None or self.eps_estimate is None or not np.isfinite([self.eps_actual, self.eps_estimate]).all():
            return "actual EPS or consensus estimate is missing"
        if not self.actual_basis or not self.estimate_basis:
            return "EPS basis (GAAP vs adjusted) not stated for the actual or the estimate"
        if self.actual_basis != self.estimate_basis:
            return f"EPS basis mismatch: actual is {self.actual_basis}, estimate is {self.estimate_basis}"
        if not self.estimate_asof:
            return "no timestamp for the consensus estimate, so it cannot be shown to predate the release"
        try:
            asof, rel = pd.Timestamp(self.estimate_asof), pd.Timestamp(self.release_date)
        except (ValueError, TypeError):
            return "unparseable release date or estimate timestamp"
        if len(self.estimate_asof.strip()) <= 10:      # date only: assume the end of that day
            asof = asof + pd.Timedelta(hours=23, minutes=59)
        if asof.tzinfo is not None:
            asof = asof.tz_convert("America/New_York").tz_localize(None)
        if asof >= (rel if self.session == "bmo" else rel + pd.Timedelta(hours=16)):
            return f"consensus estimate dated {self.estimate_asof} is not verifiably before the release"
        return None


EARNINGS_COLUMNS = ["symbol", "release_date", "session", "eps_actual", "eps_estimate", "actual_basis", "estimate_basis", "estimate_asof"]


class EarningsSource:
    def __init__(self, events=None, status: str = "not_configured", detail: str = ""):
        self._by: dict[str, list[EarningsEvent]] = {}
        for e in events or []:
            self._by.setdefault(e.symbol, []).append(e)
        self.status, self.detail = status, detail

    def events(self, symbol: str) -> list[EarningsEvent]:
        return self._by.get(symbol, [])

    def describe(self) -> dict:
        return {"status": self.status, "detail": self.detail, "events": sum(len(v) for v in self._by.values())}

    @classmethod
    def from_csv(cls, path, label: str = "") -> "EarningsSource":
        p = Path(path)
        if not p.is_file():
            return cls(status="error", detail=f"Earnings file not found: {p.name}")
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        missing = [c for c in EARNINGS_COLUMNS if rows and c not in rows[0]]
        if missing:
            return cls(status="error", detail=f"{p.name} is missing columns: {', '.join(missing)}")
        s = lambda r, k: (r.get(k) or "").strip()
        evs = [EarningsEvent(s(r, "symbol").upper(), s(r, "release_date"), s(r, "session").lower(), _num(r["eps_actual"]), _num(r["eps_estimate"]),
                             s(r, "actual_basis").lower(), s(r, "estimate_basis").lower(), s(r, "estimate_asof"), s(r, "source"))
               for r in rows if s(r, "symbol") and s(r, "release_date")]
        return cls(evs, "configured", label or f"{len(evs)} events from {p.name} (its accuracy and point-in-time integrity are yours to verify)")

    @classmethod
    def from_config(cls, cfg) -> "EarningsSource":
        c = cfg.strategies.earnings
        if not c.enabled:
            return cls(status="disabled", detail="strategies.earnings.enabled = false")
        if not c.events_csv:
            return cls(status="not_configured", detail="No earnings history configured (strategies.earnings.events_csv), so the post-earnings strategy "
                       "produces no signals. Point-in-time EPS consensus history generally requires a paid data subscription.")
        return cls.from_csv(c.events_csv)


class EconCalendar:
    """Scheduled central-bank decisions and major releases, per currency.

    CSV columns: time_utc, currency, title. The file's modification time is its "as of": if it is older
    than markets.forex.calendar_max_age_hours, the calendar is STALE and forex entries pause.
    """
    def __init__(self, events: list[tuple[pd.Timestamp, str, str]] | None = None, asof: pd.Timestamp | None = None,
                 status: str = "not_configured", detail: str = ""):
        self.rows = sorted(events or [], key=lambda r: r[0])
        self.asof, self.status, self.detail = asof, status, detail

    @classmethod
    def from_csv(cls, path, asof: pd.Timestamp | None = None) -> "EconCalendar":
        p = Path(path)
        if not p.is_file():
            return cls(status="error", detail=f"Calendar file not found: {p.name}")
        rows = []
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((pd.Timestamp(r["time_utc"], tz="UTC") if pd.Timestamp(r["time_utc"]).tzinfo is None else pd.Timestamp(r["time_utc"]).tz_convert("UTC"),
                                 r["currency"].strip().upper(), r.get("title", "").strip()))
                except (KeyError, ValueError):
                    return cls(status="error", detail=f"{p.name}: bad row {r}")
        asof = asof or pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
        return cls(rows, asof, "configured", f"{len(rows)} events from {p.name}")

    def block_reason(self, currencies: list[str], now: pd.Timestamp, before_min: float, after_min: float, max_age_hours: float) -> str | None:
        if self.status != "configured" or self.asof is None:
            return f"economic calendar unavailable ({self.status}): forex entries are paused"
        age = (now - self.asof).total_seconds() / 3600
        if age > max_age_hours:
            return f"economic calendar is stale ({age:.0f}h old, limit {max_age_hours:g}h): forex entries are paused"
        for t, ccy, title in self.rows:
            if ccy in currencies and t - pd.Timedelta(minutes=before_min) <= now <= t + pd.Timedelta(minutes=after_min):
                return f"entry blackout around {ccy} event '{title}' at {t:%Y-%m-%d %H:%M} UTC"
        return None

    def describe(self) -> dict:
        return {"status": self.status, "detail": self.detail, "asof": self.asof.isoformat() if self.asof is not None else None, "events": len(self.rows)}
