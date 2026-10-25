"""Second pass: re-fit whole wall candidates against the raw link data.

The edge-level tomography decides every 20 cm piece on its own, so a real wall
with weak evidence in places comes out broken, and a gap in a wall lets rooms
leak into each other.  Here each candidate wall (and a few metres of possible
extension along its line through unwalked floor) is cut into pieces, each piece
gets one loss value, and all of them are fitted jointly with the node offsets,
body shadowing and path-loss exponent.  Weak pieces are dropped and the fit is
repeated (backward elimination), then surviving pieces are re-joined into walls.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .data import WALKER
from .geometry import segments_intersect
from .tomography import Grid


def _pieces(seg, piece: float, max_extend: float, env, visited, g: Grid):
    """Pieces (s0, s1) along a candidate wall, including extensions through unwalked floor."""
    def cell_of(s):
        x, y = (seg.c, s) if seg.vertical else (s, seg.c)
        j, i = int((y - g.y0) / g.res), int((x - g.x0) / g.res)
        return j, i

    def state(s):
        j, i = cell_of(s)
        if not (0 <= j < g.ny and 0 <= i < g.nx) or not env[j, i]:
            return "out"
        # a wall line between two cells: walked if the cells on both sides were walked
        if seg.vertical:
            i0 = int(round((seg.c - g.x0) / g.res))
            both = visited[j, max(0, i0 - 1)] and visited[j, min(g.nx - 1, i0)]
        else:
            j0 = int(round((seg.c - g.y0) / g.res))
            both = visited[max(0, j0 - 1), i] and visited[min(g.ny - 1, j0), i]
        return "walked" if both else "free"

    L = seg.s1 - seg.s0
    k = max(1, int(round(L / piece)))
    core = [(seg.s0 + q * L / k, seg.s0 + (q + 1) * L / k) for q in range(k)]
    ext = []
    for direction in (-1, 1):
        s = seg.s0 if direction < 0 else seg.s1
        walked_run = 0
        travelled = 0.0
        while travelled < max_extend:
            a, b = (s - piece, s) if direction < 0 else (s, s + piece)
            st = state((a + b) / 2)
            if st == "out":
                break
            walked_run = walked_run + 1 if st == "walked" else 0
            if walked_run > 3:
                break
            if st == "free":
                ext.append((a, b))
            s = a if direction < 0 else b
            travelled += piece
    return sorted(core + ext)


def refine(segs, agg: dict, traj: np.ndarray, env: np.ndarray, visited: np.ndarray, g: Grid, *,
           piece: float = 0.4, max_extend: float = 2.0, keep_db: float = 1.2, smooth: float = 3.0,
           rounds: int = 3):
    """Return the candidate walls re-cut to where the joint fit supports them, with fitted losses."""
    if not segs:
        return []
    a, b, pa, pb, rssi = agg["a"], agg["b"], agg["pa"], agg["pb"], agg["rssi"]
    m = len(rssi)
    w = np.sqrt(np.minimum(agg["n"], 4.0))
    d = np.maximum(np.linalg.norm(pb - pa, axis=1), 0.3)
    nodes = sorted((set(a.tolist()) | set(b.tolist())) - {WALKER})
    ni = {k: i for i, k in enumerate(nodes)}
    walker_row = (a == WALKER) | (b == WALKER)
    O = np.zeros((m, len(nodes) + 1))
    for r in range(m):
        if walker_row[r]:
            O[r, ni[b[r] if a[r] == WALKER else a[r]]] = 1
        else:
            O[r, ni[a[r]]] += 1; O[r, ni[b[r]]] += 1; O[r, -1] = 1
    if walker_row.all():
        O = O[:, :-1]
    fixed = np.column_stack([O, -agg.get("body", np.zeros(m)), -10 * np.log10(d)])

    # candidate pieces
    P = []   # (seg index, s0, s1)
    for si, s in enumerate(segs):
        for s0, s1 in _pieces(s, piece, max_extend, env, visited, g):
            P.append((si, s0, s1))
    if not P:
        return []

    def endpoints(si, s0, s1):
        s = segs[si]
        return (np.array([s.c, s0]), np.array([s.c, s1])) if s.vertical else (np.array([s0, s.c]), np.array([s1, s.c]))

    X = np.zeros((m, len(P)))
    for k, (si, s0, s1) in enumerate(P):
        e0, e1 = endpoints(si, s0, s1)
        X[:, k] = segments_intersect(pa, pb, e0, e1)

    active = np.ones(len(P), bool)
    loss = np.zeros(len(P))
    for _ in range(rounds):
        idx = np.flatnonzero(active)
        A = np.column_stack([-X[:, idx], fixed]) * w[:, None]
        y = rssi * w
        # continuity between consecutive pieces of the same wall
        reg = []
        for q in range(len(idx) - 1):
            k0, k1 = idx[q], idx[q + 1]
            if P[k0][0] == P[k1][0] and abs(P[k1][1] - P[k0][2]) < 1e-6:
                row = np.zeros(A.shape[1]); row[q] = 1; row[q + 1] = -1
                reg.append(row)
        lam = np.sqrt(smooth * np.sum(w ** 2) / max(1, len(idx)))
        if reg:
            A = np.vstack([A, lam * np.array(reg)])
            y = np.concatenate([y, np.zeros(len(reg))])
        nf = fixed.shape[1]
        lb = np.concatenate([np.zeros(len(idx)), np.full(nf - 2, -np.inf), [0.0, 1.4]])
        ub = np.concatenate([np.full(len(idx), 30.0), np.full(nf - 2, np.inf), [15.0, 4.5]])
        sol = lsq_linear(A, y, bounds=(lb, ub), method="trf", lsmr_tol="auto").x
        loss[:] = 0
        loss[idx] = sol[:len(idx)]
        weak = active & (loss < keep_db)
        if not weak.any():
            break
        active &= ~weak

    # re-join surviving pieces into walls
    out = []
    from .extract import _Seg, _crossed  # local import to avoid a cycle
    for si, s in enumerate(segs):
        ks = [k for k, p in enumerate(P) if p[0] == si and active[k]]
        if not ks:
            continue
        runs = [[ks[0]]]
        for k in ks[1:]:
            prev = runs[-1][-1]
            gap = P[k][1] - P[prev][2]
            if gap < 1e-6 or (gap <= 1.6 and _crossed(s.vertical, s.c, P[prev][2], P[k][1], traj)):
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            s0, s1 = P[run[0]][1], P[run[-1]][2]
            out.append(_Seg(s.vertical, s.c, s0, s1, float(np.mean(loss[run]))))
    return out
