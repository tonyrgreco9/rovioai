from .alpaca import AlpacaBroker
from .base import Broker, BrokerError, BrokerTimeout, BrokerUnavailable, TERMINAL
from .oanda import OandaBroker
from .sim import SimBroker, SimWorld
from .unavailable import FIDELITY_STATUS, IBKR_STATUS

__all__ = ["Broker", "BrokerError", "BrokerTimeout", "BrokerUnavailable", "TERMINAL", "SimBroker", "SimWorld", "AlpacaBroker", "OandaBroker",
           "FIDELITY_STATUS", "IBKR_STATUS"]


def connect(cfg, live: bool = False, env=None) -> tuple[dict, dict]:
    """Build the adapters the enabled markets need. A missing credential blocks only that integration."""
    import os
    env = env if env is not None else os.environ
    brokers, status = {}, {}
    wanted = {m.broker for m in cfg.markets.values() if m.enabled}
    for key in sorted(wanted):
        try:
            if key == "alpaca":
                brokers[key] = AlpacaBroker(live=live, env=env, feed=cfg.markets.stock.data_feed)
            elif key == "oanda":
                brokers[key] = OandaBroker(live=live, env=env)
            elif key == "ibkr":
                raise BrokerUnavailable(IBKR_STATUS["summary"])
            else:
                raise BrokerUnavailable(f"unknown broker '{key}'")
            status[key] = {"available": True, "mode": brokers[key].mode}
        except BrokerUnavailable as e:
            status[key] = {"available": False, "reason": str(e)}
    return brokers, status
