"""Drive the engine through simulated time (shared by replay and backtest)."""
from __future__ import annotations

from .brokers.sim import SimBroker, SimWorld
from .engine import Engine
from .store import Store


def make(cfg, insts, exec_bars, data, mode, earnings=None, calendar=None, store=None, only=None, record_scanner=True):
    world = SimWorld(cfg, insts, exec_bars)
    keys = sorted({i.broker for i in insts})
    brokers = {k: SimBroker(k, world, cfg.portfolio.sim_starting_cash.get(k, 0.0)) for k in keys}
    store = store or Store()
    engine = Engine(cfg, store, brokers, data, mode, earnings=earnings, calendar=calendar, only=only, record_scanner=record_scanner)
    return world, brokers, store, engine


def drive(world: SimWorld, engine: Engine, times, hook=None) -> Engine:
    """`hook(t, engine)` runs before each step and may return a replacement engine (a simulated restart)."""
    for t in times:
        if hook:
            engine = hook(t, engine) or engine
        world.advance(t)
        if engine.calendar is not None and getattr(engine.calendar, "sim_always_fresh", False):
            engine.calendar.asof = t      # fixture calendars are treated as just refreshed
        engine.step(t)
    return engine
