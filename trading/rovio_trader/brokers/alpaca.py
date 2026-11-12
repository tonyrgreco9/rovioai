"""Alpaca adapter: US stocks/ETFs and spot crypto (BTC/USD).

Written against Alpaca's published Trading API v2 and Market Data API. It has NOT been exercised
against Alpaca's servers during development (no credentials, no network access there), so treat the
first paper session as the verification run: `python -m rovio_trader check alpaca`.

Paper vs live are separate hosts AND separate credential variables, so no settings change can turn a
paper session into a live one:
  paper: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY            -> paper-api.alpaca.markets
  live : ALPACA_LIVE_API_KEY_ID / ALPACA_LIVE_API_SECRET_KEY  -> api.alpaca.markets
Order types used: market (entries/exits), stop (stock protection, GTC, regular hours only) and
stop_limit (crypto protection: Alpaca crypto has no plain stop order). Stops do not trigger outside
regular hours and can gap. Fees are not reported per order, so journal fees are estimates.
"""
from __future__ import annotations

import os
import time
import urllib.parse

import pandas as pd

from ..data import daily_frame, frame
from ..instruments import ET, Instrument
from .base import Broker, BrokerError, BrokerUnavailable
from .http import Http

PAPER, LIVE, DATA = "https://paper-api.alpaca.markets", "https://api.alpaca.markets", "https://data.alpaca.markets"
STATUS = {"new": "new", "accepted": "new", "pending_new": "new", "accepted_for_bidding": "new", "held": "new", "stopped": "new", "suspended": "new",
          "calculated": "new", "pending_replace": "new", "partially_filled": "partially_filled", "filled": "filled", "canceled": "canceled",
          "replaced": "canceled", "expired": "expired", "done_for_day": "expired", "rejected": "rejected", "pending_cancel": "pending_cancel"}
CRYPTO_STOP_LIMIT_BUFFER = 0.005     # stop_limit for crypto: limit sits 0.5% beyond the stop


class AlpacaBroker(Broker):
    key = "alpaca"
    nets_positions = True

    def __init__(self, live: bool = False, env=os.environ, feed: str = "iex", opener=None, base: str | None = None, data_base: str | None = None):
        k, s = ("ALPACA_LIVE_API_KEY_ID", "ALPACA_LIVE_API_SECRET_KEY") if live else ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
        if not env.get(k) or not env.get(s):
            raise BrokerUnavailable(f"Alpaca {'live' if live else 'paper'} credentials are not set ({k} / {s} in your local .env)")
        self.mode = "LIVE" if live else "PAPER"
        self.base, self.data_base, self.feed = base or (LIVE if live else PAPER), data_base or DATA, feed
        self.http = Http({"APCA-API-KEY-ID": env[k], "APCA-API-SECRET-KEY": env[s]}, opener)
        self.data_label = (f"Alpaca {feed.upper()} feed" + (" (single venue: volume is a fraction of the consolidated tape)" if feed == "iex" else "")
                           + "; crypto: Alpaca venue quotes")
        self._clock = (0.0, None)
        self._assets: dict = {}

    def __repr__(self):
        return f"AlpacaBroker(mode={self.mode})"

    def now(self):
        return pd.Timestamp.now(tz="UTC")

    # ---- account
    def account(self):
        a = self.http.call("GET", self.base + "/v2/account")
        self.connection_verified = True
        self.account_alias = "..." + str(a.get("account_number", ""))[-4:]
        return {"equity": float(a["equity"]), "cash": float(a["cash"]), "margin_available": float(a["cash"])}

    def positions(self):
        out = {}
        for p in self.http.call("GET", self.base + "/v2/positions"):
            sym = p["symbol"]
            if p.get("asset_class") == "crypto" and "/" not in sym and sym.endswith("USD"):
                sym = f"{sym[:-3]}/USD"
            out[sym] = {"qty": float(p["qty"]), "avg_price": float(p["avg_entry_price"])}
        return out

    def instrument(self, symbol):
        if symbol not in self._assets:
            a = self.http.call("GET", self.base + "/v2/assets/" + urllib.parse.quote(symbol, safe=""), ok_404=True)
            if not a or not a.get("tradable") or a.get("status") != "active":
                self._assets[symbol] = None
            elif a.get("class") == "crypto":
                base, quote = symbol.split("/")
                self._assets[symbol] = Instrument(symbol, "crypto", self.key, tick_size=float(a.get("price_increment") or 0.01),
                                                  qty_step=float(a.get("min_trade_increment") or 0.0001), min_qty=float(a.get("min_order_size") or 0.0001),
                                                  base_ccy=base, quote_ccy=quote)
            else:
                self._assets[symbol] = Instrument(symbol, "stock", self.key, exchange=a.get("exchange", ""))   # whole shares, long only
        return self._assets[symbol]

    # ---- market state and data
    def _clk(self):
        if time.time() - self._clock[0] > 10:
            self._clock = (time.time(), self.http.call("GET", self.base + "/v2/clock"))
        return self._clock[1]

    def market_open(self, inst):
        return True if inst.market == "crypto" else bool(self._clk()["is_open"])

    def minutes_since_open(self, inst):
        if inst.market == "crypto" or not self._clk()["is_open"]:
            return None
        now = pd.Timestamp(self._clk()["timestamp"]).tz_convert(ET)
        return (now - now.normalize().replace(hour=9, minute=30)).total_seconds() / 60

    def quote(self, inst):
        if inst.market == "crypto":
            r = self.http.call("GET", self.data_base + "/v1beta3/crypto/us/latest/quotes", params={"symbols": inst.symbol})
        else:
            r = self.http.call("GET", self.data_base + "/v2/stocks/quotes/latest", params={"symbols": inst.symbol, "feed": self.feed})
        q = (r.get("quotes") or {}).get(inst.symbol)
        if not q or not q.get("ap") or not q.get("bp"):
            return None
        return {"bid": float(q["bp"]), "ask": float(q["ap"]), "ts": pd.Timestamp(q["t"]).tz_convert("UTC")}

    def bars(self, inst, timeframe, limit):
        tfmap = {"1d": "1Day", "4h": "4Hour", "1h": "1Hour", "5m": "5Min"}
        span = {"1d": pd.Timedelta(days=limit * 1.6), "4h": pd.Timedelta(hours=4 * limit), "1h": pd.Timedelta(hours=limit), "5m": pd.Timedelta(minutes=5 * limit)}[timeframe]
        params = {"symbols": inst.symbol, "timeframe": tfmap[timeframe], "start": (self.now() - span).strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "sort": "asc"}
        if inst.market == "crypto":
            url = self.data_base + "/v1beta3/crypto/us/bars"
        else:
            url, params = self.data_base + "/v2/stocks/bars", {**params, "adjustment": "all", "feed": self.feed}
        rows, token = [], None
        for _ in range(20):
            r = self.http.call("GET", url, params={**params, "page_token": token})
            rows += (r.get("bars") or {}).get(inst.symbol) or []
            token = r.get("next_page_token")
            if not token:
                break
        if not rows:
            return None
        col = lambda k: [float(x[k]) for x in rows]
        t = pd.DatetimeIndex([pd.Timestamp(x["t"]) for x in rows]).tz_convert("UTC")
        if timeframe == "1d" and inst.market == "stock":
            return daily_frame(t.tz_convert(ET).date, col("o"), col("h"), col("l"), col("c"), col("v"))
        return frame(t, t + {"4h": pd.Timedelta(hours=4), "1h": pd.Timedelta(hours=1), "5m": pd.Timedelta(minutes=5), "1d": pd.Timedelta(days=1)}[timeframe],
                     col("o"), col("h"), col("l"), col("c"), col("v"))

    # ---- orders
    @staticmethod
    def _norm(o: dict) -> dict:
        sym = o["symbol"]
        if o.get("asset_class") == "crypto" and "/" not in sym and sym.endswith("USD"):
            sym = f"{sym[:-3]}/USD"
        return {"id": o["id"], "cid": o.get("client_order_id"), "symbol": sym, "side": o["side"], "type": o.get("type") or o.get("order_type"),
                "qty": float(o.get("qty") or 0), "filled_qty": float(o.get("filled_qty") or 0),
                "avg_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None, "status": STATUS.get(o["status"], "new"),
                "stop_price": float(o["stop_price"]) if o.get("stop_price") else None, "fees": None}

    def submit(self, order):
        inst = self.instrument(order["symbol"])
        if inst is None:
            raise BrokerError(f"{order['symbol']} is not tradable at Alpaca")
        if order["side"] == "sell":
            # Alpaca has no reduce-only flag: check the position ourselves so an exit can never open a short.
            held = self.positions().get(inst.symbol, {}).get("qty", 0.0)
            if order["qty"] > held + 1e-9:
                raise BrokerError(f"sell of {order['qty']:g} exceeds the {held:g} held; refusing to open a short")
        qty = f"{order['qty']:.9f}".rstrip("0").rstrip(".")
        body = {"symbol": inst.symbol, "qty": qty, "side": order["side"], "client_order_id": order["cid"]}
        crypto = inst.market == "crypto"
        if order["type"] == "market":
            body.update(type="market", time_in_force="gtc" if crypto else "day")
        elif crypto:
            lim = order["stop_price"] * (1 - CRYPTO_STOP_LIMIT_BUFFER if order["side"] == "sell" else 1 + CRYPTO_STOP_LIMIT_BUFFER)
            body.update(type="stop_limit", time_in_force="gtc", stop_price=str(inst.round_price(order["stop_price"])), limit_price=str(inst.round_price(lim)))
        else:
            body.update(type="stop", time_in_force="gtc", stop_price=f"{order['stop_price']:.2f}")
        return self._norm(self.http.call("POST", self.base + "/v2/orders", body))

    def get_by_cid(self, cid):
        o = self.http.call("GET", self.base + "/v2/orders:by_client_order_id", params={"client_order_id": cid}, ok_404=True)
        return self._norm(o) if o else None

    def cancel(self, order):
        try:
            self.http.call("DELETE", self.base + "/v2/orders/" + order["id"], ok_404=True)
        except BrokerError as e:
            if "422" not in str(e):      # 422 = no longer cancelable (already filled or cancelled)
                raise

    def open_orders(self):
        return [self._norm(o) for o in self.http.call("GET", self.base + "/v2/orders", params={"status": "open", "limit": 500})]
