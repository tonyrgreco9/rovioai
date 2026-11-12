"""OANDA v20 adapter: spot forex.

Written against OANDA's published v20 REST API. It has NOT been exercised against OANDA's servers
during development, so treat the first Practice session as the verification run:
`python -m rovio_trader check oanda`.

  practice: OANDA_API_TOKEN / OANDA_ACCOUNT_ID            -> api-fxpractice.oanda.com
  live    : OANDA_LIVE_API_TOKEN / OANDA_LIVE_ACCOUNT_ID  -> api-fxtrade.oanda.com
Order types used: MARKET (fill-or-kill) for entries and exits, STOP (GTC) for protection. Exits and
stops are sent with positionFill=REDUCE_ONLY, so they can never open or reverse a position.
The account must be denominated in USD (the portfolio risk view is USD).
"""
from __future__ import annotations

import os

import pandas as pd

from ..data import frame
from ..instruments import Instrument
from .base import Broker, BrokerError, BrokerUnavailable
from .http import Http

PRACTICE, LIVE = "https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"
GRAN = {"1h": "H1", "4h": "H4", "5m": "M5", "1d": "D"}
SPAN = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "5m": pd.Timedelta(minutes=5), "1d": pd.Timedelta(days=1)}


class OandaBroker(Broker):
    key = "oanda"
    nets_positions = True
    data_label = "OANDA executable bid/ask (dealer quotes, not an exchange tape)"

    def __init__(self, live: bool = False, env=os.environ, opener=None, base: str | None = None):
        t, a = ("OANDA_LIVE_API_TOKEN", "OANDA_LIVE_ACCOUNT_ID") if live else ("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID")
        if not env.get(t) or not env.get(a):
            raise BrokerUnavailable(f"OANDA {'live' if live else 'practice'} credentials are not set ({t} / {a} in your local .env)")
        self.mode = "LIVE" if live else "PAPER"
        self.http = Http({"Authorization": "Bearer " + env[t], "Accept-Datetime-Format": "RFC3339"}, opener)
        self._acct = (base or (LIVE if live else PRACTICE)) + "/v3/accounts/" + env[a]
        self._root = base or (LIVE if live else PRACTICE)
        self.account_alias = "..." + env[a][-4:]
        self._inst: dict = {}

    def __repr__(self):
        return f"OandaBroker(mode={self.mode})"

    @staticmethod
    def _sym(symbol: str) -> str:
        return symbol.replace("/", "_")

    def now(self):
        return pd.Timestamp.now(tz="UTC")

    def account(self):
        a = self.http.call("GET", self._acct + "/summary")["account"]
        if a.get("currency") != "USD":
            raise BrokerUnavailable(f"OANDA account currency is {a.get('currency')}; this bot requires a USD account")
        self.connection_verified = True
        return {"equity": float(a["NAV"]), "cash": float(a["balance"]), "margin_available": float(a["marginAvailable"])}

    def positions(self):
        out = {}
        for p in self.http.call("GET", self._acct + "/openPositions").get("positions", []):
            net = float(p["long"]["units"]) + float(p["short"]["units"])
            side = p["long"] if net > 0 else p["short"]
            if net:
                out[p["instrument"].replace("_", "/")] = {"qty": net, "avg_price": float(side.get("averagePrice") or 0)}
        return out

    def instrument(self, symbol):
        if symbol not in self._inst:
            r = self.http.call("GET", self._acct + "/instruments", params={"instruments": self._sym(symbol)}).get("instruments", [])
            if not r:
                self._inst[symbol] = None
            else:
                i = r[0]
                base, quote = symbol.split("/")
                self._inst[symbol] = Instrument(symbol, "forex", self.key, tick_size=10.0 ** -int(i["displayPrecision"]), pip=10.0 ** int(i["pipLocation"]),
                                                qty_step=10.0 ** -int(i.get("tradeUnitsPrecision", 0)), min_qty=float(i.get("minimumTradeSize", 1)),
                                                base_ccy=base, quote_ccy=quote, shortable=True, margin_rate=float(i.get("marginRate", 0.05)))
        return self._inst[symbol]

    def _price(self, inst):
        p = self.http.call("GET", self._acct + "/pricing", params={"instruments": self._sym(inst.symbol)}).get("prices", [])
        return p[0] if p else None

    def market_open(self, inst):
        p = self._price(inst)
        return bool(p and p.get("tradeable"))

    def quote(self, inst):
        p = self._price(inst)
        if not p or not p.get("bids") or not p.get("asks"):
            return None
        return {"bid": float(p["bids"][0]["price"]), "ask": float(p["asks"][0]["price"]), "ts": pd.Timestamp(p["time"]).tz_convert("UTC")}

    def bars(self, inst, timeframe, limit):
        r = self.http.call("GET", f"{self._root}/v3/instruments/{self._sym(inst.symbol)}/candles",
                           params={"granularity": GRAN[timeframe], "count": min(limit, 5000), "price": "M"})
        c = [x for x in r.get("candles", []) if x.get("complete")]
        if not c:
            return None
        t = pd.DatetimeIndex([pd.Timestamp(x["time"]) for x in c]).tz_convert("UTC")
        col = lambda k: [float(x["mid"][k]) for x in c]
        return frame(t, t + SPAN[timeframe], col("o"), col("h"), col("l"), col("c"), [float(x.get("volume", 0)) for x in c])

    # ---- orders
    def _fill(self, tx_id) -> tuple[float, float, float]:
        tx = self.http.call("GET", f"{self._acct}/transactions/{tx_id}")["transaction"]
        return abs(float(tx["units"])), float(tx["price"]), abs(float(tx.get("commission", 0))) + max(-float(tx.get("financing", 0)), 0.0)

    def _norm(self, o: dict) -> dict:
        state, units = o.get("state"), float(o.get("units", 0))
        out = {"id": o["id"], "cid": (o.get("clientExtensions") or {}).get("id"), "symbol": o.get("instrument", "").replace("_", "/"),
               "side": "buy" if units > 0 else "sell", "type": "stop" if o.get("type") == "STOP" else "market", "qty": abs(units), "filled_qty": 0.0,
               "avg_price": None, "status": {"PENDING": "new", "TRIGGERED": "new", "CANCELLED": "canceled", "FILLED": "filled"}.get(state, "new"),
               "stop_price": float(o["price"]) if o.get("price") else None, "fees": None}
        if state == "FILLED" and o.get("fillingTransactionID"):
            out["filled_qty"], out["avg_price"], out["fees"] = self._fill(o["fillingTransactionID"])
        return out

    def submit(self, order):
        units = order["qty"] * (1 if order["side"] == "buy" else -1)
        body = {"instrument": self._sym(order["symbol"]), "units": f"{units:.0f}", "clientExtensions": {"id": order["cid"]},
                "positionFill": "REDUCE_ONLY" if order.get("reduce_only") else "DEFAULT"}
        if order["type"] == "market":
            body.update(type="MARKET", timeInForce="FOK")
        else:
            inst = self.instrument(order["symbol"])
            body.update(type="STOP", timeInForce="GTC", price=f"{order['stop_price']:.{max(len(f'{inst.tick_size:.10f}'.rstrip('0').split('.')[1]), 1)}f}")
        r = self.http.call("POST", self._acct + "/orders", {"order": body})
        if "orderRejectTransaction" in r:
            raise BrokerError(f"rejected: {r['orderRejectTransaction'].get('rejectReason', 'unknown')}")
        created = r.get("orderCreateTransaction", {})
        out = {"id": created.get("id"), "cid": order["cid"], "symbol": order["symbol"], "side": order["side"], "type": order["type"], "qty": order["qty"],
               "filled_qty": 0.0, "avg_price": None, "status": "new", "stop_price": order.get("stop_price"), "fees": None}
        if "orderFillTransaction" in r:
            f = r["orderFillTransaction"]
            out.update(filled_qty=abs(float(f["units"])), avg_price=float(f["price"]), status="filled", fees=abs(float(f.get("commission", 0))))
        elif "orderCancelTransaction" in r:
            out.update(status="canceled", note=r["orderCancelTransaction"].get("reason"))
        return out

    def get_by_cid(self, cid):
        r = self.http.call("GET", f"{self._acct}/orders/@{cid}", ok_404=True)
        return self._norm(r["order"]) if r and r.get("order") else None

    def cancel(self, order):
        try:
            self.http.call("PUT", f"{self._acct}/orders/@{order['cid']}/cancel", ok_404=True)
        except BrokerError:
            pass                              # already filled or cancelled; the next read shows which

    def open_orders(self):
        return [self._norm(o) for o in self.http.call("GET", self._acct + "/pendingOrders").get("orders", [])]
