"""Ties data, signals, risk and a broker together: scan -> propose -> (approve) -> execute."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import risk, signals
from .brokers import AlpacaBroker, Broker, BrokerError, ManualBroker, Order, PaperBroker
from .brokers.base import now_iso
from .config import Settings
from .data import DataError, Provider, SyntheticProvider, YahooProvider, asset_class, normalize_symbol

NOTES = {
    "mutual_fund": "Mutual funds trade once a day at the closing NAV, and many charge fees or restrict accounts for round trips inside 30-90 days. Treat signals as slow allocation guidance, not trade timing.",
    "forex": "Quantities are units of the base currency; spreads, leverage and financing are not modelled.",
    "crypto": "Trades 24/7 with high volatility; expect large gaps relative to the ATR stop.",
}


class RiskRejected(Exception):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def build(settings: Settings) -> "Engine":
    provider: Provider = SyntheticProvider() if settings.demo else YahooProvider()
    tag = "demo-" if settings.demo else ""
    if settings.broker == "alpaca":
        if settings.demo:
            raise ValueError("Demo mode uses fake prices and cannot be combined with the Alpaca broker.")
        broker: Broker = AlpacaBroker()
    elif settings.broker == "manual":
        broker = ManualBroker(settings.state_dir / f"{tag}manual.json", provider.last_price, settings.starting_cash)
    else:
        broker = PaperBroker(settings.state_dir / f"{tag}paper.json", provider.last_price, settings.starting_cash)
    return Engine(settings, provider, broker)


class Engine:
    def __init__(self, settings: Settings, provider: Provider, broker: Broker):
        self.s, self.provider, self.broker = settings, provider, broker
        self._lock = threading.RLock()
        self._path = Path(settings.state_dir) / f"{'demo-' if settings.demo else ''}engine-{broker.name}.json"
        self.state = {"watchlist": [normalize_symbol(x) for x in settings.watchlist], "proposals": [], "kill_switch": False, "day": None}
        if self._path.exists():
            self.state.update(json.loads(self._path.read_text()))

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self._path)

    # ---- configuration -------------------------------------------------
    def info(self) -> dict:
        return {"broker": self.broker.name, "live": self.broker.live, "executes": self.broker.executes, "demo": self.s.demo,
                "data": self.provider.name, "kill_switch": self.state["kill_switch"], "watchlist": self.state["watchlist"],
                "risk": vars(self.s.risk)}

    def set_kill_switch(self, on: bool):
        with self._lock:
            self.state["kill_switch"] = bool(on)
            self._save()

    def edit_watchlist(self, add: str | None = None, remove: str | None = None) -> list[str]:
        with self._lock:
            wl = self.state["watchlist"]
            if add:
                sym = normalize_symbol(add)
                self.provider.history(sym, "1mo", "1d")  # reject symbols with no data
                if sym not in wl:
                    wl.append(sym)
            if remove and normalize_symbol(remove) in wl:
                wl.remove(normalize_symbol(remove))
            self._save()
            return wl

    # ---- analysis ------------------------------------------------------
    def analyze(self, symbol: str, range_: str = "2y", interval: str = "1d") -> dict:
        symbol = normalize_symbol(symbol)
        out = signals.analyze(self.provider.history(symbol, range_, interval), self.s.risk.atr_stop_mult)
        cls = asset_class(symbol)
        out.update(symbol=symbol, asset_class=cls, note=NOTES.get(cls))
        return out

    def scan(self) -> list[dict]:
        rows = []
        for sym in list(self.state["watchlist"]):
            try:
                rows.append(self.analyze(sym))
            except (DataError, ValueError) as e:
                rows.append({"symbol": sym, "asset_class": asset_class(sym), "error": str(e)})
        return rows

    # ---- portfolio / orders --------------------------------------------
    def portfolio(self) -> dict:
        acct = self.broker.account()
        return {"account": acct, "positions": self.broker.positions(), "orders": self.broker.orders()[:50],
                "day_start_equity": self._day_start(acct["equity"])}

    def _day_start(self, equity: float) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if not self.state["day"] or self.state["day"]["date"] != today:
                self.state["day"] = {"date": today, "equity": equity}
                self._save()
            return self.state["day"]["equity"]

    def submit_order(self, symbol: str, side: str, qty: float, note: str = "") -> dict:
        """Risk-check and send one market order. Raises RiskRejected with every violated rule."""
        symbol, side, qty = normalize_symbol(symbol), side.lower(), float(qty)
        price = self.provider.last_price(symbol)
        acct = self.broker.account()
        held = {p["symbol"]: p["qty"] for p in self.broker.positions()}
        errs = risk.check(self.s.risk, side=side, symbol=symbol, qty=qty, price=price, equity=acct["equity"], cash=acct["cash"],
                          positions=held, kill_switch=self.state["kill_switch"], day_start_equity=self._day_start(acct["equity"]))
        unsupported = self.broker.supports(symbol)
        if unsupported:
            errs.append(unsupported)
        if errs:
            raise RiskRejected(errs)
        return self.broker.submit(Order(symbol=symbol, side=side, qty=qty, note=note)).to_dict()

    # ---- proposals -----------------------------------------------------
    def generate_proposals(self) -> list[dict]:
        """Turn the current scan into sized order proposals. Nothing is sent to the broker."""
        acct = self.broker.account()
        held = {p["symbol"]: p["qty"] for p in self.broker.positions()}
        cash, open_n, fresh = acct["cash"], sum(1 for q in held.values() if q > 0), []
        for a in sorted((r for r in self.scan() if "error" not in r and r["action"] != "NO DATA"), key=lambda r: -r["score"]):
            sym, why = a["symbol"], "; ".join(c["reason"] for c in a["components"] if abs(c["points"] or 0) >= 10)
            side = qty = None
            if held.get(sym, 0) > 0 and a["score"] < 0:
                side, qty = "sell", held[sym]
            elif held.get(sym, 0) <= 0 and a["score"] >= signals.BUY and open_n < self.s.risk.max_open_positions and not self.broker.supports(sym):
                qty = risk.position_size(self.s.risk, sym, acct["equity"], cash, a["price"], a["levels"]["atr"])
                if qty > 0:
                    side, cash, open_n = "buy", cash - qty * a["price"], open_n + 1
            if side:
                fresh.append({"id": uuid.uuid4().hex[:10], "symbol": sym, "side": side, "qty": qty, "est_price": a["price"],
                              "est_value": round(qty * a["price"], 2), "score": a["score"], "action": a["action"], "reason": why,
                              "stop": a["levels"]["stop"] if side == "buy" else None, "note": a.get("note"),
                              "status": "pending", "created": now_iso()})
        with self._lock:
            done = [p for p in self.state["proposals"] if p["status"] != "pending"][-50:]
            self.state["proposals"] = done + fresh
            self._save()
        return fresh

    def proposals(self) -> list[dict]:
        ps = self.state["proposals"]  # pending first (best score first), then history newest first
        return [p for p in ps if p["status"] == "pending"] + [p for p in reversed(ps) if p["status"] != "pending"]

    def _decide(self, pid: str, status: str, **extra) -> dict:
        with self._lock:
            for p in self.state["proposals"]:
                if p["id"] == pid:
                    if p["status"] != "pending":
                        raise BrokerError(f"Proposal {pid} is already {p['status']}")
                    p.update(status=status, decided=now_iso(), **extra)
                    self._save()
                    return p
        raise BrokerError(f"No such proposal {pid}")

    def approve(self, pid: str) -> dict:
        with self._lock:
            p = next((x for x in self.state["proposals"] if x["id"] == pid and x["status"] == "pending"), None)
            if not p:
                raise BrokerError(f"No pending proposal {pid}")
            order = self.submit_order(p["symbol"], p["side"], p["qty"], note=f"Signal {p['action']} ({p['score']:+g})")
            return self._decide(pid, "executed", order_id=order["id"])

    def reject(self, pid: str) -> dict:
        return self._decide(pid, "rejected")

    def run_cycle(self, auto: bool = False) -> dict:
        """One scan. With auto=True, approved automatically (sells first); otherwise proposals wait for you."""
        props, results = self.generate_proposals(), []
        if auto:
            for p in sorted(props, key=lambda p: p["side"] != "sell"):
                try:
                    self.approve(p["id"])
                    results.append({"symbol": p["symbol"], "side": p["side"], "qty": p["qty"], "ok": True})
                except (RiskRejected, BrokerError, DataError) as e:
                    results.append({"symbol": p["symbol"], "side": p["side"], "qty": p["qty"], "ok": False, "error": str(e)})
        return {"proposals": props, "executed": results}
