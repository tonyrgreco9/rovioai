"""Manual-ticket broker, for accounts with no trading API (e.g. Fidelity retail accounts).

`submit` does NOT trade. It records an order ticket for you to enter at your broker yourself;
once you report the actual fill price, the tracked portfolio is updated."""
from __future__ import annotations

from .base import BrokerError, Order
from .paper import PaperBroker


class ManualBroker(PaperBroker):
    name = "manual"
    executes = False

    def submit(self, order: Order) -> Order:
        order.status = "ticket"
        order.note = (order.note + " " if order.note else "") + "Place this order at your broker, then record the fill."
        with self._lock:
            self.state["orders"].append(order.to_dict())
            self._save()
        return order

    def _find(self, order_id: str) -> dict:
        for o in self.state["orders"]:
            if o["id"] == order_id:
                if o["status"] != "ticket":
                    raise BrokerError(f"Order {order_id} is {o['status']}, not an open ticket")
                return o
        raise BrokerError(f"No such order {order_id}")

    def confirm_fill(self, order_id: str, price: float, qty: float | None = None) -> dict:
        if price <= 0:
            raise BrokerError("Fill price must be positive")
        with self._lock:
            rec = self._find(order_id)
            order = Order(**rec)
            if qty:
                order.qty = float(qty)
            self._apply_fill(order, float(price))
            rec.update(order.to_dict())
            self._save()
            return rec

    def cancel(self, order_id: str) -> dict:
        with self._lock:
            rec = self._find(order_id)
            rec["status"] = "cancelled"
            self._save()
            return rec
