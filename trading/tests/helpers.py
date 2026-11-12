import pandas as pd

from rovio_trader import config, fixtures as F, simrun

U = lambda s: pd.Timestamp(s, tz="UTC")


def world_for(symbols, overrides=None, mode="REPLAY", store=None):
    """A small simulated world built from the replay fixtures, restricted to `symbols`."""
    ov = {k: dict(v) for k, v in F.OVERRIDES.items()}
    ov["markets"] = {"stock": {"watchlist": [s for s in symbols if s.startswith("RPL")], "etfs": ["RPLX"]}}
    for k, v in (overrides or {}).items():
        ov.setdefault(k, {}).update(v)
    cfg = config.load(None, ov)
    insts, ex, data, earn, cal = F.build(cfg)
    insts = [i for i in insts if i.symbol in symbols]
    world, brokers, store, eng = simrun.make(cfg, insts, {s: ex[s] for s in symbols}, data, mode, earn, cal, store=store)
    return cfg, world, brokers, store, eng, cal


def run(world, eng, start, end, cal=None, each=None):
    for t in world.timeline(U(start), U(end)):
        world.advance(t)
        if cal is not None and getattr(cal, "sim_always_fresh", False):
            cal.asof = t
        eng.step(t)
        if each:
            each(t)
    return eng
