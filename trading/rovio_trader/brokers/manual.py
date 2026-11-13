"""Order tickets for an account with no order-entry API (Fidelity retail).

Nothing here contacts Fidelity. `submit` writes a ticket for YOU to place by hand; the ticket stays open
until you record what happened (placed, filled at a price, or not done). The engine then treats your
recorded fills exactly like broker fills: it tracks the position, issues the protective-stop ticket and
later the exit ticket. Prices come from a separate public feed (see yahoo.py).

Because there is no connection, the account figures are bookkeeping: the cash you said the bot may use,
adjusted by the fills you recorded. A fill you forget to record makes them wrong.
"""
from __future__ import annotations

import pandas as pd

from ..instruments import stock
from .base import TERMINAL, Broker, BrokerError
from .yahoo import LABEL, YahooFeed

OPEN = ("new", "partially_filled")
EPS = 1e-9
SETTLE = pd.Timedelta(minutes=20)       # a daily bar counts as complete this long after the close (delayed closing prints)


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def ticket_action(store, cid: str, action: str, qty=None, price=None, fees=0.0) -> dict:
    """What the account holder did with a ticket. Used by the dashboard and the command line."""
    with store.tx():
        t = store.one("SELECT * FROM tickets WHERE cid=?", (cid,))
        if t is None:
            raise ValueError("No such ticket")
        if t["status"] not in OPEN:
            raise ValueError(f"This ticket is already {t['status']}")
        if action == "placed":
            store.update("tickets", "cid", cid, placed=1, updated=_now())
        elif action == "fill":
            try:
                qty, price, fees = float(qty), float(price), float(fees or 0)
            except (TypeError, ValueError):
                raise ValueError("Enter the filled quantity and the fill price as numbers") from None
            left = t["qty"] - t["filled_qty"]
            if not (qty > 0 and price > 0 and fees >= 0) or qty != qty or price != price:
                raise ValueError("Quantity and price must be greater than zero")
            if qty > left + EPS:
                raise ValueError(f"Only {left:g} remain on this ticket; enter what actually filled (at most {left:g})")
            done = t["filled_qty"] + qty
            avg = ((t["avg_price"] or 0.0) * t["filled_qty"] + price * qty) / done
            store.update("tickets", "cid", cid, filled_qty=done, avg_price=avg, fees=(t["fees"] or 0.0) + fees, placed=1,
                         status="filled" if done >= t["qty"] - EPS else "partially_filled", updated=_now())
        elif action == "cancel":
            store.update("tickets", "cid", cid, status="canceled", updated=_now(),
                         note="cancel confirmed by you" if t["cancel_requested"] else "not placed / cancelled by you")
        else:
            raise ValueError("action must be placed, fill or cancel")
    return store.one("SELECT * FROM tickets WHERE cid=?", (cid,))


def instruction(t: dict) -> str:
    """Plain-language text for one ticket."""
    left = t["qty"] - t["filled_qty"]
    verb = t["side"].upper()
    if t["cancel_requested"]:
        return (f"CANCEL your open {t['type']} order to {t['side']} {left:g} {t['symbol']} at Fidelity, then confirm here. "
                f"If it already executed, record the fill instead.")
    if t["type"] == "stop":
        if t["placed"]:
            return f"Resting at Fidelity: {verb} STOP {left:g} {t['symbol']} at {t['stop_price']:g}. If it triggers, record the fill here."
        return (f"Place a {verb} STOP order at Fidelity, good 'til cancelled: {left:g} shares of {t['symbol']}, stop price {t['stop_price']:g}. "
                f"Then mark it placed. Until you do, this position has NO protective stop.")
    why = {"entry": "new position", "exit": "close the position"}.get(t["role"], t["role"])
    return f"{verb} {left:g} shares of {t['symbol']} at market ({why}). When it fills, record the quantity and price here."


class ManualBroker(Broker):
    key = "fidelity"
    mode = "MANUAL"
    manual = True                # orders are tickets a person places; nothing is transmitted
    account_alias = "tickets only - not connected"
    data_label = LABEL

    def __init__(self, store, feed=None, clock=None):
        self.store, self.feed, self._clock = store, feed or YahooFeed(), clock
        self._last: dict[str, float] = {}

    # ---- clock, account, positions
    def now(self):
        return self._clock() if self._clock else pd.Timestamp.now(tz="UTC")

    def _fills(self):
        return self.store.q("SELECT symbol, side, filled_qty, avg_price, fees FROM tickets WHERE filled_qty>0")

    def positions(self):
        pos: dict[str, dict] = {}
        for f in self._fills():
            p = pos.setdefault(f["symbol"], {"qty": 0.0, "cost": 0.0, "bought": 0.0})
            sign = 1 if f["side"] == "buy" else -1
            p["qty"] += sign * f["filled_qty"]
            if sign > 0:
                p["cost"] += f["filled_qty"] * f["avg_price"]
                p["bought"] += f["filled_qty"]
        return {s: {"qty": round(p["qty"], 8), "avg_price": p["cost"] / p["bought"] if p["bought"] else None} for s, p in pos.items() if abs(p["qty"]) > 1e-8}

    def account(self):
        start = self.store.get("manual_cash")
        if start is None:
            raise BrokerError("the cash amount for ticket mode is not set (run: worker --manual --cash AMOUNT)")
        cash = float(start)
        for f in self._fills():
            cash += (-1 if f["side"] == "buy" else 1) * f["filled_qty"] * f["avg_price"] - (f["fees"] or 0.0)
        pos = self.positions()
        equity = cash + sum(p["qty"] * (self._last.get(s) or p["avg_price"] or 0.0) for s, p in pos.items())
        return {"equity": equity, "cash": cash, "margin_available": max(cash, 0.0)}

    # ---- instruments and prices
    def _chart(self, symbol, tf):
        c = self.feed.chart(symbol, tf)
        if c and c["meta"].get("regularMarketPrice"):
            self._last[symbol] = float(c["meta"]["regularMarketPrice"])
        return c

    def instrument(self, symbol):
        c = self._chart(symbol, "1d")
        if c is None or c["meta"].get("instrumentType") not in ("EQUITY", "ETF") or (c["meta"].get("currency") or "USD") != "USD":
            return None
        return stock(symbol, self.key, is_etf=c["meta"].get("instrumentType") == "ETF")

    def _session(self, inst):
        c = self._chart(inst.symbol, "5m")
        reg = ((c or {}).get("meta", {}).get("currentTradingPeriod") or {}).get("regular") or {}
        if not reg.get("start") or not reg.get("end"):
            return None
        return pd.Timestamp(reg["start"], unit="s", tz="UTC"), pd.Timestamp(reg["end"], unit="s", tz="UTC"), c["meta"]

    def market_open(self, inst):
        s = self._session(inst)
        if s is None:
            return False
        start, end, meta = s
        last = pd.Timestamp(meta.get("regularMarketTime") or 0, unit="s", tz="UTC")
        return bool(start <= self.now() < end and last >= start)      # a trade today: rules out holidays the feed did not flag

    def minutes_since_open(self, inst):
        s = self._session(inst)
        return (self.now() - s[0]).total_seconds() / 60 if s and s[0] <= self.now() < s[1] else None

    def quote(self, inst):
        c = self._chart(inst.symbol, "5m")
        m = (c or {}).get("meta") or {}
        if not m.get("regularMarketPrice") or not m.get("regularMarketTime"):
            return None
        p = float(m["regularMarketPrice"])
        return {"bid": p, "ask": p, "ts": pd.Timestamp(m["regularMarketTime"], unit="s", tz="UTC"), "last_only": True}

    def bars(self, inst, timeframe, limit):
        c = self._chart(inst.symbol, timeframe)
        if c is None or not len(c["bars"]):
            return None
        df = c["bars"]
        if timeframe == "1d" and self.now() - SETTLE < df.index[-1] <= self.now():
            df = df.iloc[:-1]
        return df.iloc[-limit:]

    # ---- tickets
    @staticmethod
    def _norm(t: dict) -> dict:
        return {"id": t["cid"], "cid": t["cid"], "symbol": t["symbol"], "side": t["side"], "type": t["type"], "qty": t["qty"], "filled_qty": t["filled_qty"],
                "avg_price": t["avg_price"], "status": t["status"], "stop_price": t["stop_price"], "fees": t["fees"] or 0.0, "note": t["note"]}

    def submit(self, order):
        ts = self.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.store.insert("tickets", ignore=True, cid=order["cid"], symbol=order["symbol"], side=order["side"], type=order["type"], qty=order["qty"],
                          stop_price=order.get("stop_price"), role=order.get("role") or "", status="new", created=ts, updated=ts)
        return self.get_by_cid(order["cid"])

    def get_by_cid(self, cid):
        t = self.store.one("SELECT * FROM tickets WHERE cid=?", (cid,))
        return self._norm(t) if t else None

    def cancel(self, order):
        t = self.store.one("SELECT * FROM tickets WHERE cid=?", (order["cid"],))
        if t is None or t["status"] in TERMINAL:
            return
        if not t["placed"]:
            # you never said it was placed, so there is nothing at Fidelity to cancel
            self.store.update("tickets", "cid", t["cid"], status="canceled", note="withdrawn before you placed it", updated=_now())
        else:
            self.store.update("tickets", "cid", t["cid"], cancel_requested=1, updated=_now())

    def open_orders(self):
        return [self._norm(t) for t in self.store.q("SELECT * FROM tickets WHERE status IN ('new','partially_filled')")]

    def describe(self):
        return {**super().describe(), "manual": True, "feed_verified": bool(getattr(self.feed, "verified", False))}
