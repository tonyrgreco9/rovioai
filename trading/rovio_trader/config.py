"""Settings from environment variables (optionally loaded from a local, gitignored .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .risk import RiskConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = "SPY,QQQ,AAPL,MSFT,NVDA,BTC-USD,ETH-USD,EURUSD=X,FXAIX"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing variables. Values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    broker: str = "paper"
    demo: bool = False
    watchlist: list[str] = field(default_factory=lambda: DEFAULT_WATCHLIST.split(","))
    state_dir: Path = ROOT / ".state"
    starting_cash: float = 100_000.0
    risk: RiskConfig = field(default_factory=RiskConfig)
    host: str = "127.0.0.1"
    port: int = 8787

    @classmethod
    def from_env(cls, env=os.environ, **over) -> "Settings":
        f = lambda k, d: float(env.get(k) or d)
        s = cls(
            broker=(env.get("ROVIO_BROKER") or "paper").strip().lower(),
            watchlist=[x.strip() for x in (env.get("ROVIO_WATCHLIST") or DEFAULT_WATCHLIST).split(",") if x.strip()],
            starting_cash=f("ROVIO_STARTING_CASH", 100_000),
            risk=RiskConfig(
                risk_per_trade_pct=f("ROVIO_RISK_PER_TRADE_PCT", 1.0),
                max_position_pct=f("ROVIO_MAX_POSITION_PCT", 20),
                max_open_positions=int(f("ROVIO_MAX_OPEN_POSITIONS", 10)),
                max_daily_loss_pct=f("ROVIO_MAX_DAILY_LOSS_PCT", 3.0),
            ),
        )
        for k, v in over.items():
            if v is not None:
                setattr(s, k, v)
        if s.broker not in ("paper", "manual", "alpaca"):
            raise ValueError(f"ROVIO_BROKER must be paper, manual or alpaca (got {s.broker!r})")
        return s
