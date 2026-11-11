"""Alpaca Trading API adapter (US stocks/ETFs and crypto).

Credentials are read from the environment (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY, normally
set in your local, gitignored .env). They are only ever sent to Alpaca in request headers and are
never logged, returned, or written to disk by this code. The paper-trading endpoint is used
unless ALPACA_LIVE is exactly "true".
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..data import asset_class
from .base import Broker, BrokerError, Order

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


def _to_alpaca(symbol: str) -> str:
    return symbol.replace("-", "/") if asset_class(symbol) == "crypto" else symbol


def _from_alpaca(symbol: str, cls: str) -> str:
    if cls == "crypto":
        s = symbol.replace("/", "")
        return f"{s[:-3]}-{s[-3:]}" if s.endswith("USD") else symbol
    return symbol


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, env=os.environ, opener=urllib.request.urlopen):
        self._key, self._secret = env.get("ALPACA_API_KEY_ID", ""), env.get("ALPACA_API_SECRET_KEY", "")
        if not self._key or not self._secret:
            raise BrokerError("Alpaca credentials are not set. Add them to your local .env (see .env.example).")
        self.live = env.get("ALPACA_LIVE", "").strip().lower() == "true"
        self.base = LIVE_URL if self.live else PAPER_URL
        self._open = opener

    def __repr__(self):  # never expose credentials
        return f"AlpacaBroker(live={self.live})"

    def _call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret, "Content-Type": "application/json"},
        )
        try:
            with self._open(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                msg = json.load(e).get("message", "")
            except Exception:
                msg = ""
            raise BrokerError(f"Alpaca rejected the request ({e.code}): {msg}") from None
        except Exception as e:
            raise BrokerError(f"Alpaca request failed: {type(e).__name__}") from None

    def supports(self, symbol):
        cls = asset_class(symbol)
        if cls in ("forex", "mutual_fund"):
            return f"Alpaca does not trade {cls.replace('_', ' ')}s."
        return None

    def account(self):
        a = self._call("GET", "/v2/account")
        return {"cash": float(a["cash"]), "equity": float(a["equity"])}

    def positions(self):
        out = []
        for p in self._call("GET", "/v2/positions"):
            qty, avg, px = float(p["qty"]), float(p["avg_entry_price"]), float(p.get("current_price") or p["avg_entry_price"])
            out.append({"symbol": _from_alpaca(p["symbol"], p.get("asset_class", "")), "qty": qty, "avg_price": avg,
                        "price": px, "value": qty * px, "pnl": (px - avg) * qty})
        return out

    def orders(self):
        rows = self._call("GET", "/v2/orders?status=all&limit=50")
        return [{"id": o["id"], "symbol": _from_alpaca(o["symbol"], o.get("asset_class", "")), "side": o["side"],
                 "qty": float(o.get("qty") or 0), "type": o.get("type", ""), "status": o["status"], "created": o.get("created_at"),
                 "filled_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                 "filled_at": o.get("filled_at"), "note": ""} for o in rows]

    def submit(self, order: Order) -> Order:
        reason = self.supports(order.symbol)
        if reason:
            raise BrokerError(reason)
        tif = "gtc" if asset_class(order.symbol) == "crypto" else "day"
        res = self._call("POST", "/v2/orders", {"symbol": _to_alpaca(order.symbol), "qty": f"{order.qty:.9f}".rstrip("0").rstrip("."), "side": order.side,
                                                "type": "market", "time_in_force": tif, "client_order_id": order.id})
        order.id, order.status = res.get("id", order.id), "submitted"
        return order
