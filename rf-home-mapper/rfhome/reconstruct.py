"""End-to-end pipeline: survey -> aligned grid -> thin-wall tomography -> floor plan."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Survey, aggregate
from .extract import ExtractParams, extract
from .geometry import Door, FloorPlan, Room, Wall
from .tomography import Grid, RTIResult, dominant_angle, solve


@dataclass
class Params:
    res: float = 0.2              # grid cell (m); wall positions are quantised to this
    cell: float = 0.25            # averaging bin for walker positions (m)
    angle: float | None = None    # house orientation (rad); estimated from the walk if None
    height: float = 2.4
    refine: bool = True           # second-pass whole-wall fit
    solver: dict = field(default_factory=lambda: {"l1": 0.05, "smooth": 50.0})
    extract: ExtractParams = field(default_factory=ExtractParams)


@dataclass
class Result:
    plan: FloorPlan
    rti: RTIResult
    angle: float
    survey_aligned: Survey


def reconstruct(s: Survey, p: Params | None = None) -> Result:
    p = p or Params()
    theta = dominant_angle(s.traj[:, 1:3]) if p.angle is None else p.angle
    sa = s.rotated(-theta)                              # work in a house-aligned frame
    traj = sa.traj[:, 1:3]
    pts = [traj, sa.pa, sa.pb]
    if sa.nodes:
        pts.append(np.array(list(sa.nodes.values()), float))
    grid = Grid.around(np.vstack(pts), p.res, 1.0)
    agg = aggregate(sa, p.cell)
    rti = solve(agg, traj, grid, **p.solver)
    plan = extract(rti, traj, sa.nodes, p.extract, p.height, agg if p.refine else None)
    return Result(rotate_plan(plan, theta), rti, theta, sa)


def rotate_plan(plan: FloorPlan, theta: float) -> FloorPlan:
    if theta == 0:
        return plan
    c, s = np.cos(theta), np.sin(theta)
    rot = lambda x, y: (float(c * x - s * y), float(s * x + c * y))
    walls = [Wall(*rot(w.x0, w.y0), *rot(w.x1, w.y1), w.loss_db, w.exterior) for w in plan.walls]
    rooms = [Room(r.name, [rot(*q) for q in r.polygon], r.area_m2) for r in plan.rooms]
    return FloorPlan(walls, [Door(d.wall, d.center, d.width) for d in plan.doors], rooms, plan.height)
