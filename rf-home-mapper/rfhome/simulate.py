"""Synthetic home + survey generator.

Used to test the reconstruction end to end against a known ground truth before
pointing it at a real house.  The radio model is the standard indoor
log-distance model with per-wall attenuation (multi-wall / COST-231 style):

    rssi = P1m + offset(node) - 10 n log10(d) - sum(wall losses crossed)
           - body loss + correlated shadowing + fast fading

It is deliberately harsher than a textbook simulation (body shadowing,
spatially correlated shadowing, integer RSSI, receiver sensitivity floor), but
real homes are still messier: furniture, mirrors, appliances and people all
change the numbers.  Treat simulated accuracy as an upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .data import WALKER, Survey
from .geometry import Door, FloorPlan, Room, Wall, segments_intersect


@dataclass
class RectRoom:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float


def example_home() -> tuple[FloorPlan, list[RectRoom], list[tuple[str, str, tuple[float, float]]],
                            dict[str, tuple[float, float]]]:
    """An 11 x 8 m flat: living room, kitchen, two bedrooms, bathroom, hallway."""
    rooms = [
        RectRoom("stue", 0, 0, 6, 5),
        RectRoom("kjokken", 6, 0, 11, 5),
        RectRoom("soverom", 0, 5, 4, 8),
        RectRoom("bad", 4, 5, 6, 8),
        RectRoom("gang", 6, 5, 7.5, 8),
        RectRoom("soverom2", 7.5, 5, 11, 8),
    ]
    walls = [
        Wall(0, 0, 11, 0, 12, True), Wall(11, 0, 11, 8, 12, True),
        Wall(11, 8, 0, 8, 12, True), Wall(0, 8, 0, 0, 12, True),
        Wall(0, 5, 11, 5, 5),          # 4: long interior wall
        Wall(6, 0, 6, 5, 5),           # 5: living | kitchen
        Wall(4, 5, 4, 8, 5),           # 6: bedroom | bath
        Wall(6, 5, 6, 8, 8),           # 7: bath | hall (tiled, heavier)
        Wall(7.5, 5, 7.5, 8, 5),       # 8: hall | bedroom 2
    ]
    doors = [
        Door(4, 2.0, 0.9),     # living -> bedroom
        Door(4, 6.75, 0.9),    # kitchen -> hall
        Door(5, 3.0, 1.2),     # living <-> kitchen opening
        Door(7, 1.5, 0.8),     # hall -> bath
        Door(8, 1.5, 0.9),     # hall -> bedroom 2
    ]
    # (from_room, to_room, door centre) used to plan the walk
    links = [
        ("stue", "soverom", (2.0, 5.0)),
        ("stue", "kjokken", (6.0, 3.0)),
        ("kjokken", "gang", (6.75, 5.0)),
        ("gang", "bad", (6.0, 6.5)),
        ("gang", "soverom2", (7.5, 6.5)),
    ]
    nodes = {
        "stue": (0.3, 0.3),
        "stue2": (5.7, 4.7),
        "kjokken": (10.7, 0.3),
        "soverom": (0.3, 7.7),
        "bad": (5.0, 7.7),
        "soverom2": (10.7, 7.7),
    }
    plan = FloorPlan(walls, doors, [Room(r.name, [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)],
                                         (r.x1 - r.x0) * (r.y1 - r.y0)) for r in rooms])
    return plan, rooms, links, nodes


def walk_path(rooms: list[RectRoom], links, start: str, margin: float = 0.45, spacing: float = 0.8) -> np.ndarray:
    """Depth-first tour that lawn-mows every room and passes through doors."""
    by_name = {r.name: r for r in rooms}
    adj: dict[str, list[tuple[str, tuple[float, float]]]] = {}
    for a, b, d in links:
        adj.setdefault(a, []).append((b, d))
        adj.setdefault(b, []).append((a, d))
    pts: list[tuple[float, float]] = []
    seen = set()

    def inside(r: RectRoom, p):
        return (min(max(p[0], r.x0 + margin), r.x1 - margin), min(max(p[1], r.y0 + margin), r.y1 - margin))

    def sweep(r: RectRoom):
        xs = np.arange(r.x0 + margin, r.x1 - margin + 1e-9, spacing)
        if len(xs) == 0:
            xs = np.array([(r.x0 + r.x1) / 2])
        for k, x in enumerate(xs):
            ys = (r.y0 + margin, r.y1 - margin)
            pts.extend([(x, ys[k % 2]), (x, ys[(k + 1) % 2])])

    def visit(name: str):
        seen.add(name)
        r = by_name[name]
        sweep(r)
        for other, door in adj.get(name, []):
            if other in seen:
                continue
            pts.extend([inside(r, door), door])
            visit(other)
            pts.extend([door, inside(r, door)])

    visit(start)
    return np.array(pts)


def resample(waypoints: np.ndarray, speed: float, rate: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ts = np.arange(0, s[-1] / speed, 1 / rate)
    d = ts * speed
    return np.column_stack([ts, np.interp(d, s, waypoints[:, 0]), np.interp(d, s, waypoints[:, 1])])


def _shadow_field(rng, sigma: float, n_waves: int = 12):
    k = rng.uniform(2 * np.pi / 3.0, 2 * np.pi / 0.8, n_waves)
    ang = rng.uniform(0, 2 * np.pi, n_waves)
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    kv = np.column_stack([k * np.cos(ang), k * np.sin(ang)])
    amp = sigma * np.sqrt(2 / n_waves)
    return lambda p: amp * np.cos(p @ kv.T + ph).sum(axis=1)


def path_loss(plan: FloorPlan, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Total wall loss (dB) on each straight path pa[i] -> pb[i]."""
    loss = np.zeros(len(pa))
    for _, w in plan.solid_pieces():
        hit = segments_intersect(pa, pb, np.array([w.x0, w.y0]), np.array([w.x1, w.y1]))
        loss += hit * w.loss_db
    return loss


def simulate_survey(plan: FloorPlan, rooms, links, nodes, seed: int = 0, *, start: str | None = None,
                    p1m: float = -50.0, n_exp: float = 2.2, fading_sigma: float = 3.0,
                    shadow_sigma: float = 2.0, body_loss: float = 4.0, pos_sigma: float = 0.15,
                    rate: float = 4.0, speed: float = 0.8, floor_dbm: float = -98.0) -> Survey:
    rng = np.random.default_rng(seed)
    traj = resample(walk_path(rooms, links, start or rooms[0].name), speed, rate)
    true_xy = traj[:, 1:3]
    # what the walker *thinks* their position is (AR tracking / step counting is not perfect)
    # tracking error is smooth (AR/VIO drifts slowly), not white jitter
    drift = np.cumsum(rng.normal(0, 0.004, true_xy.shape), axis=0)
    wobble = ndimage.gaussian_filter1d(rng.normal(0, 1, true_xy.shape), sigma=rate * 2, axis=0)
    wobble *= pos_sigma / max(wobble.std(), 1e-9)
    est_xy = true_xy + drift + wobble
    heading = np.arctan2(*np.gradient(true_xy, axis=0)[:, ::-1].T)

    offs = {k: rng.uniform(-3, 3) for k in nodes}
    shadow = {k: _shadow_field(rng, shadow_sigma) for k in nodes}
    t_all, a_all, b_all, r_all, pa_all, pb_all = [], [], [], [], [], []

    for nid, pos in nodes.items():
        pn = np.tile(np.asarray(pos, float), (len(traj), 1))
        d = np.maximum(np.linalg.norm(true_xy - pn, axis=1), 0.3)
        to_node = np.arctan2(pn[:, 1] - true_xy[:, 1], pn[:, 0] - true_xy[:, 0])
        behind = np.abs(np.angle(np.exp(1j * (to_node - heading)))) > np.deg2rad(110)
        rssi = (p1m + offs[nid] - 10 * n_exp * np.log10(d) - path_loss(plan, true_xy, pn)
                - body_loss * behind + shadow[nid](true_xy) + rng.normal(0, fading_sigma, len(traj)))
        rssi = np.round(rssi)
        ok = rssi > floor_dbm
        t_all.append(traj[ok, 0]); r_all.append(rssi[ok])
        a_all.append(np.full(ok.sum(), WALKER)); b_all.append(np.full(ok.sum(), nid))
        pa_all.append(est_xy[ok]); pb_all.append(pn[ok])

    # node <-> node beacons, one per second in each direction
    ids = list(nodes)
    tn = np.arange(0, traj[-1, 0], 1.0)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa = np.tile(nodes[a], (len(tn), 1)).astype(float)
            pb = np.tile(nodes[b], (len(tn), 1)).astype(float)
            d = np.linalg.norm(pa - pb, axis=1)
            base = (p1m + offs[a] + offs[b] - 10 * n_exp * np.log10(d) - path_loss(plan, pa, pb))
            for src, dst, P, Q in ((a, b, pa, pb), (b, a, pb, pa)):
                rssi = np.round(base + rng.normal(0, fading_sigma, len(tn)))
                ok = rssi > floor_dbm
                t_all.append(tn[ok]); r_all.append(rssi[ok])
                a_all.append(np.full(ok.sum(), src)); b_all.append(np.full(ok.sum(), dst))
                pa_all.append(P[ok]); pb_all.append(Q[ok])

    order = np.argsort(np.concatenate(t_all), kind="stable")
    cat = lambda xs: np.concatenate(xs)[order]
    return Survey(cat(t_all), cat(a_all), cat(b_all), cat(r_all), cat(pa_all), cat(pb_all),
                  np.column_stack([traj[:, 0], est_xy]), {k: tuple(map(float, v)) for k, v in nodes.items()})
