from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BrokerError(Exception):
    pass


@dataclass
class Order:
    symbol: str
    side: str            # buy | sell
    qty: float
    type: str = "market"
    status: str = "new"  # new | filled | ticket (awaiting manual entry) | submitted | rejected | cancelled
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = field(default_factory=now_iso)
    filled_price: float | None = None
    filled_at: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker:
    """Minimal broker interface. Implementations must never log or return credentials."""
    name = "base"
    live = False              # True only when real money is at risk
    executes = True           # False when a human must place the order (manual tickets)

    def account(self) -> dict:           # {"cash": float, "equity": float}
        raise NotImplementedError

    def positions(self) -> list[dict]:   # [{"symbol","qty","avg_price","price","value","pnl"}]
        raise NotImplementedError

    def orders(self) -> list[dict]:
        raise NotImplementedError

    def submit(self, order: Order) -> Order:
        raise NotImplementedError

    def supports(self, symbol: str) -> str | None:
        """Return a reason string if this broker cannot trade the symbol, else None."""
        return None
