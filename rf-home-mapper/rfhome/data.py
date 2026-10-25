"""Survey data: radio links and the walker's trajectory.

Files on disk
-------------
``measurements.csv``  one row per received packet::

    t,a,b,rssi,ax,ay,bx,by

  ``a``/``b`` are device ids (``walker`` for the device you carry, otherwise the
  ESP32 node id).  ``ax..by`` are the positions of both ends when the packet was
  received.

``trajectory.csv``  where the walker was::

    t,x,y

``nodes.json``  fixed receivers ``{"kitchen": [x, y], ...}``.  Node ids are
  usually room names in ESPresense, which is reused to name the rooms found.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WALKER = "walker"


@dataclass
class Survey:
    t: np.ndarray          # (M,)
    a: np.ndarray          # (M,) str
    b: np.ndarray          # (M,) str
    rssi: np.ndarray       # (M,)
    pa: np.ndarray         # (M,2)
    pb: np.ndarray         # (M,2)
    traj: np.ndarray       # (T,3) t,x,y
    nodes: dict[str, tuple[float, float]]

    def __len__(self) -> int:
        return len(self.rssi)

    @property
    def node_ids(self) -> list[str]:
        ids = set(self.a.tolist()) | set(self.b.tolist()) | set(self.nodes)
        ids.discard(WALKER)
        return sorted(ids)

    def rotated(self, theta: float) -> "Survey":
        """Copy with every position rotated by `theta` radians about the origin."""
        R = _rot(theta)
        return Survey(
            self.t, self.a, self.b, self.rssi, self.pa @ R.T, self.pb @ R.T,
            np.column_stack([self.traj[:, 0], self.traj[:, 1:3] @ R.T]),
            {k: tuple(np.asarray(v) @ R.T) for k, v in self.nodes.items()},
        )

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "measurements.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "a", "b", "rssi", "ax", "ay", "bx", "by"])
            for i in range(len(self)):
                w.writerow([f"{self.t[i]:.3f}", self.a[i], self.b[i], f"{self.rssi[i]:.1f}",
                            *(f"{v:.3f}" for v in (*self.pa[i], *self.pb[i]))])
        with open(folder / "trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y"])
            for row in self.traj:
                w.writerow([f"{v:.3f}" for v in row])
        (folder / "nodes.json").write_text(json.dumps({k: list(map(float, v)) for k, v in self.nodes.items()}, indent=2))

    @classmethod
    def load(cls, folder: str | Path) -> "Survey":
        folder = Path(folder)
        nodes = {}
        if (folder / "nodes.json").exists():
            nodes = {k: tuple(v) for k, v in json.loads((folder / "nodes.json").read_text()).items()}
        cols = {k: [] for k in ["t", "a", "b", "rssi", "ax", "ay", "bx", "by"]}
        with open(folder / "measurements.csv", newline="") as f:
            for row in csv.DictReader(f):
                for k in cols:
                    cols[k].append(row[k])
        traj = np.loadtxt(folder / "trajectory.csv", delimiter=",", skiprows=1, ndmin=2) \
            if (folder / "trajectory.csv").exists() else np.zeros((0, 3))
        f = lambda k: np.asarray(cols[k], dtype=float)
        return cls(
            t=f("t"), a=np.asarray(cols["a"]), b=np.asarray(cols["b"]), rssi=f("rssi"),
            pa=np.column_stack([f("ax"), f("ay")]), pb=np.column_stack([f("bx"), f("by")]),
            traj=traj, nodes=nodes,
        )


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_factor(s: Survey, min_speed: float = 0.2, window: float = 1.0) -> np.ndarray:
    """How much the walker's own body sits between the carried device and the other end.

    ``max(0, -cos(angle between walking direction and direction to the node))``:
    0 when walking towards the node, 1 when walking straight away from it.
    People hold the phone in front of them, so a node behind them is shadowed by
    5-10 dB of body.  Without this term the alternating directions of walking
    lanes show up as phantom walls between the lanes.  0 while standing still.
    """
    f = np.zeros(len(s))
    if len(s.traj) < 3:
        return f
    tt, xy = s.traj[:, 0], s.traj[:, 1:3]
    half = window / 2
    is_a = s.a == WALKER
    is_b = s.b == WALKER
    rows = np.flatnonzero(is_a | is_b)
    if len(rows) == 0:
        return f
    t = s.t[rows]
    p0 = np.column_stack([np.interp(t - half, tt, xy[:, 0]), np.interp(t - half, tt, xy[:, 1])])
    p1 = np.column_stack([np.interp(t + half, tt, xy[:, 0]), np.interp(t + half, tt, xy[:, 1])])
    v = p1 - p0
    speed = np.linalg.norm(v, axis=1) / window
    me = np.where(is_a[rows, None], s.pa[rows], s.pb[rows])
    other = np.where(is_a[rows, None], s.pb[rows], s.pa[rows])
    to = other - me
    cos = np.einsum("ij,ij->i", v, to) / np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(to, axis=1), 1e-9)
    f[rows] = np.where(speed >= min_speed, np.maximum(0.0, -cos), 0.0)
    return f


def aggregate(s: Survey, cell: float = 0.25) -> dict[str, np.ndarray]:
    """Average packets that share a link and (for walker links) a `cell`-sized position bin.

    Raw RSSI has several dB of fast fading; averaging packets taken within a few
    decimetres of each other suppresses that without blurring walls much.
    Returns arrays keyed like the Survey plus ``n`` (packets per row) and
    ``body`` (mean body-shadowing factor, see `body_factor`).
    """
    # links are reciprocal: order every row so that a <= b before grouping
    body = body_factor(s)
    swap = s.a > s.b
    A = np.where(swap, s.b, s.a)
    B = np.where(swap, s.a, s.b)
    PA = np.where(swap[:, None], s.pb, s.pa)
    PB = np.where(swap[:, None], s.pa, s.pb)
    key_a = np.floor(PA / cell).astype(int)
    key_b = np.floor(PB / cell).astype(int)
    groups: dict[tuple, list[int]] = {}
    for i in range(len(s)):
        groups.setdefault((A[i], B[i], *key_a[i], *key_b[i]), []).append(i)
    idx = list(groups.values())
    first = np.array([g[0] for g in idx], dtype=int)
    return {
        "a": A[first], "b": B[first],
        "pa": np.array([PA[g].mean(0) for g in idx]).reshape(-1, 2),
        "pb": np.array([PB[g].mean(0) for g in idx]).reshape(-1, 2),
        "rssi": np.array([np.median(s.rssi[g]) for g in idx]),
        "n": np.array([len(g) for g in idx]),
        "body": np.array([body[g].mean() for g in idx]),
    }
