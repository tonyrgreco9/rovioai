"""Configuration: a TOML file for settings, environment variables (local .env) for credentials.

Every strategy number below is an INITIAL RESEARCH SETTING: a starting point for testing, not a
validated parameter. The risk numbers are PAPER-MODE EXAMPLES; live trading refuses to start until
you set your own (risk.set_by_user = true). Override anything in your config file, not here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "state_dir": "state",
    "poll_seconds": 20,
    "alert_webhook_env": "ROVIO_ALERT_WEBHOOK",   # NAME of an env var holding a webhook URL (optional)
    "portfolio": {
        "daily_reset_tz": "America/New_York",
        "daily_reset_time": "17:00",              # daily loss counters reset here every day, weekends included
        "sim_starting_cash": {"alpaca": 100000.0, "oanda": 50000.0, "ibkr": 50000.0},  # replay/backtest only
    },
    "markets": {
        "stock": {"enabled": True, "broker": "alpaca", "watchlist": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"],
                  "etfs": ["SPY", "QQQ", "IWM", "DIA"], "data_feed": "iex"},
        "futures": {"enabled": True, "broker": "ibkr", "roots": ["MES", "MNQ"], "expiry_exclusion_days": 7},
        "forex": {"enabled": True, "broker": "oanda", "pairs": ["EUR/USD", "GBP/USD", "USD/JPY"], "calendar_csv": "", "calendar_max_age_hours": 24.0},
        "crypto": {"enabled": True, "broker": "alpaca", "symbols": ["BTC/USD"]},
    },
    "strategies": {
        "breakout": {"enabled": True, "lookback": 20, "volume_mult": 1.5, "volume_lookback": 20, "sma_period": 50, "atr_period": 14,
                     "stop_atr_mult": 2.0, "exit_channel": 10, "max_hold_sessions": 30, "max_gap_pct": 3.0, "max_spread_bps": 20.0,
                     "entry_delay_min": 5.0, "entry_cutoff_min": 60.0},
        "earnings": {"enabled": True, "events_csv": "", "min_surprise_pct": 0.0, "volume_mult": 1.5, "volume_lookback": 20, "atr_period": 14,
                     "min_stop_atr": 1.0, "max_hold_sessions": 40, "max_gap_pct": 3.0, "max_spread_bps": 20.0,
                     "entry_delay_min": 5.0, "entry_cutoff_min": 60.0},
        "orb": {"enabled": True, "range_minutes": 30, "bar_minutes": 5, "trend_filter": "ema", "trend_ema": 20, "atr_period": 14,
                "stop_atr_mult": 3.0, "target_r": 2.0, "last_entry": "14:30", "flatten": "15:45", "allow_short": True,
                "max_trades_per_day": 1, "max_spread_ticks": 2.0, "entry_valid_min": 5.0, "max_slippage_atr": 1.0},
        "fx_pullback": {"enabled": True, "fast_ema": 20, "slow_ema": 50, "pullback_ema": 20, "atr_period": 14, "stop_atr_mult": 2.0,
                        "target_r": 2.0, "max_hold_hours": 72.0, "allow_short": True, "max_spread_pips": 2.5,
                        "blackout_before_min": 60.0, "blackout_after_min": 30.0, "entry_valid_min": 15.0, "max_slippage_atr": 0.5},
        "btc_breakout": {"enabled": True, "lookback": 20, "daily_trend_sma": 20, "atr_period": 14, "stop_atr_mult": 2.5, "exit_channel": 10,
                         "max_hold_bars": 30, "max_spread_bps": 30.0, "entry_valid_min": 30.0, "max_slippage_atr": 0.5},
    },
    "risk": {   # PAPER-MODE EXAMPLE LIMITS
        "set_by_user": False,
        "max_loss_per_trade_pct": 0.5,      # planned loss at the stop, % of total equity
        "max_loss_per_trade_usd": 0.0,      # optional dollar cap (0 = off)
        "max_open_risk_pct": 3.0,           # planned loss summed over open positions and pending entries
        "max_gross_exposure_pct": 200.0,    # gross notional / equity (200 = 2x leverage)
        "max_instrument_pct": 25.0,         # notional in one instrument
        "max_positions": 8,
        "max_order_notional_usd": 50000.0,
        "max_volume_participation": 0.01,   # liquidity: order size vs the signal bar's volume (where volume exists)
        "daily_loss_limit_pct": 2.0,        # realized + unrealized, since the daily reset
        "max_drawdown_pct": 10.0,           # peak-to-trough, realized + unrealized
        "max_quote_age_sec": 30.0,
        "emergency_exit_policy": "flatten", # when protection cannot be established: flatten | hold_and_alert
        "per_market_pct": {"stock": 100.0, "futures": 100.0, "forex": 150.0, "crypto": 20.0},
        "per_broker_pct": {"alpaca": 100.0, "oanda": 150.0, "ibkr": 100.0},
        "max_currency_exposure_pct": 150.0, # net exposure to any single non-USD currency
        "groups": {"sp500": {"symbols": ["SPY", "MES"], "max_pct": 30.0}, "nasdaq100": {"symbols": ["QQQ", "MNQ"], "max_pct": 30.0}},
    },
    "costs": {  # simulation assumptions for replay and backtest; set them to what your broker actually charges
        "stock": {"half_spread_bps": 2.5, "slippage_bps": 5.0, "fee_per_unit": 0.0, "fee_bps": 0.0},
        "futures": {"half_spread_ticks": 0.5, "slippage_ticks": 0.5, "fee_per_unit": 0.62, "fee_bps": 0.0},
        "forex": {"half_spread_pips": 0.6, "slippage_pips": 0.2, "fee_per_unit": 0.0, "fee_bps": 0.0, "financing_bps_per_day": 0.8},
        "crypto": {"half_spread_bps": 5.0, "slippage_bps": 5.0, "fee_per_unit": 0.0, "fee_bps": 25.0},
    },
    "evidence": {"train_fraction": 0.5, "validation_fraction": 0.25, "min_oos_trades": 30, "min_t_stat": 2.0},
}

STRATEGY_MARKET = {"breakout": "stock", "earnings": "stock", "orb": "futures", "fx_pullback": "forex", "btc_breakout": "crypto"}
STRATEGY_VERSION = {"breakout": "1.0", "earnings": "1.0", "orb": "1.0", "fx_pullback": "1.0", "btc_breakout": "1.0"}
OPEN_TABLES = {("portfolio", "sim_starting_cash"), ("risk", "per_market_pct"), ("risk", "per_broker_pct"), ("risk", "groups")}


class Cfg(dict):
    """Dict with attribute access (cfg.risk.max_positions)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    @property
    def state_path(self) -> Path:
        p = Path(self["state_dir"])
        return p if p.is_absolute() else ROOT / p

    def strategy_hash(self, name: str) -> str:
        """Fingerprint of what determines a strategy's trades; evidence and live activation are tied to it."""
        risk = {k: v for k, v in self["risk"].items() if k != "set_by_user"}
        blob = {"v": STRATEGY_VERSION[name], "p": self["strategies"][name], "risk": risk, "market": self["markets"][STRATEGY_MARKET[name]]}
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


def _wrap(d):
    return Cfg({k: _wrap(v) for k, v in d.items()}) if isinstance(d, dict) else d


def _merge(base: dict, over: dict, path: tuple):
    for k, v in over.items():
        if path in OPEN_TABLES:
            base[k] = v
            continue
        if k not in base:
            raise ValueError(f"Unknown setting {'.'.join(path + (k,))!r}")
        cur = base[k]
        if isinstance(cur, dict):
            if not isinstance(v, dict):
                raise ValueError(f"{'.'.join(path + (k,))} must be a table")
            _merge(cur, v, path + (k,))
        elif isinstance(cur, bool) != isinstance(v, bool) or (isinstance(cur, (int, float)) and not isinstance(v, (int, float))) \
                or (isinstance(cur, (str, list)) and not isinstance(v, type(cur))):
            raise ValueError(f"{'.'.join(path + (k,))} has the wrong type (expected {type(cur).__name__})")
        else:
            base[k] = float(v) if isinstance(cur, float) else v


def load(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    data = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "rb") as f:
            _merge(data, tomllib.load(f), ())
    if overrides:
        _merge(data, overrides, ())
    r = data["risk"]
    for k in ("max_loss_per_trade_pct", "max_open_risk_pct", "daily_loss_limit_pct", "max_drawdown_pct"):
        if not 0 < r[k] <= 100:
            raise ValueError(f"risk.{k} must be between 0 and 100")
    if r["emergency_exit_policy"] not in ("flatten", "hold_and_alert"):
        raise ValueError("risk.emergency_exit_policy must be 'flatten' or 'hold_and_alert'")
    return _wrap(data)


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing variables. Values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SECRET_ENV = ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "ALPACA_LIVE_API_KEY_ID", "ALPACA_LIVE_API_SECRET_KEY",
              "OANDA_API_TOKEN", "OANDA_ACCOUNT_ID", "OANDA_LIVE_API_TOKEN", "OANDA_LIVE_ACCOUNT_ID", "ROVIO_ALERT_WEBHOOK")


def redact(text, env=os.environ) -> str:
    """Mask any credential value that appears in a string bound for a log, the database or the screen."""
    out = str(text)
    for k in SECRET_ENV:
        v = env.get(k)
        if v and len(v) >= 4:
            out = out.replace(v, "[REDACTED]")
    return out
