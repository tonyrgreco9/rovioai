from __future__ import annotations


class BrokerError(Exception):
    """The broker definitively refused or failed the request (nothing was accepted)."""


class BrokerTimeout(Exception):
    """Outcome unknown: the request may or may not have been accepted. Reconcile before retrying."""


class BrokerUnavailable(Exception):
    """This integration cannot be used (missing credentials, unsupported, or not implemented)."""


TERMINAL = ("filled", "canceled", "rejected", "expired")


class Broker:
    """Adapter interface. One instance = one account at one broker.

    Normalised order dict returned by submit/get_by_cid/open_orders:
      {"id", "cid", "symbol", "side", "type", "qty", "filled_qty", "avg_price", "status", "stop_price", "fees"}
    status is one of: new | partially_filled | filled | canceled | rejected | expired | pending_cancel
    `fees` is None when the broker does not report them per order.
    Adapters must never log, return or store credentials.
    """
    key = "base"                 # config name: alpaca | oanda | ibkr
    mode = "PAPER"               # REPLAY | PAPER | LIVE
    account_alias = ""
    data_label = ""              # e.g. "IEX real-time (single venue)", "SIMULATED fixture"
    connection_verified = False  # True only after a real successful broker response in this process
    nets_positions = True        # broker keeps one net position per instrument

    def now(self): raise NotImplementedError
    def account(self) -> dict: raise NotImplementedError            # {"equity","cash","margin_available"} in USD
    def positions(self) -> dict: raise NotImplementedError          # {symbol: {"qty": signed, "avg_price"}}
    def instrument(self, symbol: str): raise NotImplementedError    # Instrument as the broker defines it, or None if not tradable
    def contracts(self, root: str) -> list: return []               # futures: dated contracts for a root
    def market_open(self, inst) -> bool: raise NotImplementedError
    def minutes_since_open(self, inst): return None                 # None when the venue has no session open (24h) or it is unknown
    def quote(self, inst) -> dict | None: raise NotImplementedError # {"bid","ask","ts"} or None
    def bars(self, inst, timeframe: str, limit: int): raise NotImplementedError
    def submit(self, order: dict) -> dict: raise NotImplementedError
    def get_by_cid(self, cid: str) -> dict | None: raise NotImplementedError
    def cancel(self, order: dict) -> None: raise NotImplementedError
    def open_orders(self) -> list[dict]: raise NotImplementedError
    def describe(self) -> dict:
        return {"key": self.key, "mode": self.mode, "account": self.account_alias, "data": self.data_label, "verified": self.connection_verified}
