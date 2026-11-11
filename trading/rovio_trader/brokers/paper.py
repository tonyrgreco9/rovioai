"""Simulated broker: cash account, long-only, market orders filled at the latest price plus slippage.
State is a local JSON file. No real money moves."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .base import Broker, BrokerError, Order, now_iso


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, state_file: Path, price_fn, starting_cash: float = 100_000.0, slippage_bps: float = 5.0):
        self.path, self.price_fn, self.slip = Path(state_file), price_fn, slippage_bps / 1e4
        self._lock = threading.RLock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": float(starting_cash), "starting_cash": float(starting_cash), "positions": {}, "orders": []}
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.path)

    def positions(self):
        out = []
        with self._lock:
            items = list(self.state["positions"].items())
        for sym, p in items:
            try:
                px = self.price_fn(sym)
            except Exception:
                px = p["avg_price"]  # price unavailable: carry at cost rather than crash
            out.append({"symbol": sym, "qty": p["qty"], "avg_price": p["avg_price"], "price": px,
                        "value": p["qty"] * px, "pnl": (px - p["avg_price"]) * p["qty"]})
        return out

    def account(self):
        value = sum(p["value"] for p in self.positions())
        with self._lock:
            cash = self.state["cash"]
        return {"cash": cash, "equity": cash + value, "starting_cash": self.state.get("starting_cash")}

    def orders(self):
        with self._lock:
            return list(reversed(self.state["orders"]))

    def _apply_fill(self, order: Order, price: float):
        """Book a fill. Caller holds the lock."""
        pos = self.state["positions"].get(order.symbol, {"qty": 0.0, "avg_price": 0.0})
        if order.side == "buy":
            cost = order.qty * price
            if cost > self.state["cash"] + 1e-6:
                raise BrokerError("Insufficient cash")
            self.state["cash"] -= cost
            new_qty = pos["qty"] + order.qty
            pos = {"qty": new_qty, "avg_price": (pos["qty"] * pos["avg_price"] + cost) / new_qty}
            self.state["positions"][order.symbol] = pos
        else:
            if order.qty > pos["qty"] + 1e-9:
                raise BrokerError("Insufficient position")
            self.state["cash"] += order.qty * price
            left = pos["qty"] - order.qty
            if left <= 1e-9:
                self.state["positions"].pop(order.symbol, None)
            else:
                self.state["positions"][order.symbol] = {"qty": left, "avg_price": pos["avg_price"]}
        order.status, order.filled_price, order.filled_at = "filled", price, now_iso()

    def submit(self, order: Order) -> Order:
        px = self.price_fn(order.symbol) * (1 + self.slip if order.side == "buy" else 1 - self.slip)
        with self._lock:
            self._apply_fill(order, px)
            self.state["orders"].append(order.to_dict())
            self._save()
        return order
