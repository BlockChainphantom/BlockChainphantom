"""Floor-plan primitives: walls, doors, rooms and the plan that holds them.

All coordinates are metres in a right-handed frame (x east, y north).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Wall:
    x0: float
    y0: float
    x1: float
    y1: float
    loss_db: float = 0.0      # attenuation of one crossing at 2.4 GHz
    exterior: bool = False

    @property
    def length(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    def points(self, step: float) -> np.ndarray:
        """Points sampled every `step` metres along the wall (endpoints included)."""
        n = max(2, int(np.ceil(self.length / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        return np.column_stack([self.x0 + t * (self.x1 - self.x0), self.y0 + t * (self.y1 - self.y0)])


@dataclass
class Door:
    wall: int                 # index into FloorPlan.walls
    center: float             # distance from the wall start (m)
    width: float = 0.9


@dataclass
class Room:
    name: str
    polygon: list[tuple[float, float]]
    area_m2: float


@dataclass
class FloorPlan:
    walls: list[Wall] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    height: float = 2.4

    def to_json(self) -> str:
        return json.dumps(
            {
                "height": self.height,
                "walls": [asdict(w) for w in self.walls],
                "doors": [asdict(d) for d in self.doors],
                "rooms": [asdict(r) for r in self.rooms],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "FloorPlan":
        d = json.loads(text)
        return cls(
            walls=[Wall(**w) for w in d.get("walls", [])],
            doors=[Door(**x) for x in d.get("doors", [])],
            rooms=[Room(name=r["name"], polygon=[tuple(p) for p in r["polygon"]], area_m2=r["area_m2"])
                   for r in d.get("rooms", [])],
            height=d.get("height", 2.4),
        )

    def solid_pieces(self) -> list[tuple[int, Wall]]:
        """Walls split at door openings, i.e. the parts a radio wave actually passes through."""
        out = []
        for i, w in enumerate(self.walls):
            L = w.length
            if L == 0:
                continue
            gaps = sorted((d.center - d.width / 2, d.center + d.width / 2) for d in self.doors if d.wall == i)
            start = 0.0
            for g0, g1 in gaps + [(L, L)]:
                g0, g1 = max(0.0, g0), min(L, g1)
                if g0 > start:
                    out.append((i, _sub(w, start / L, g0 / L)))
                start = max(start, g1)
        return out


def _sub(w: Wall, t0: float, t1: float) -> Wall:
    dx, dy = w.x1 - w.x0, w.y1 - w.y0
    return Wall(w.x0 + t0 * dx, w.y0 + t0 * dy, w.x0 + t1 * dx, w.y0 + t1 * dy, w.loss_db, w.exterior)


def segments_intersect(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised proper-intersection test of segments p->q (N,2) against a single segment a->b."""
    def cross(o, u, v):
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])

    d1 = cross(a, b, p)
    d2 = cross(a, b, q)
    d3 = cross(p, q, a)
    d4 = cross(p, q, b)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab) or 1e-12
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
