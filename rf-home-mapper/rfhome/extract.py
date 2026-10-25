"""Turn the edge-loss map + walked area into walls, doors and rooms.

Everything here works in the house-aligned grid frame, so walls are axis
aligned (the Manhattan assumption).  `reconstruct.py` rotates the result back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure, segmentation

from .geometry import Door, FloorPlan, Room, Wall, segments_intersect
from .refine import refine
from .tomography import Grid, RTIResult


@dataclass
class ExtractParams:
    edge_min: float = 0.3        # dB; edges below this are treated as empty
    merge_dist: float = 0.6      # m; parallel strands closer than this are one wall
    min_len: float = 0.8         # m; shortest interior wall kept
    min_loss: float = 2.0        # dB per crossing; weakest interior wall kept
    gap_join: float = 0.4        # m; collinear walls with a smaller gap are joined
    door_join: float = 1.4       # m; ... or with a larger gap if the walker went through it
    snap: float = 0.8            # m; how far wall ends are extended to meet another wall
    envelope_close: float = 0.8  # m; closes gaps between walking lanes
    clearance: float = 0.35      # m; how close to walls you walked (outer walls sit this far out)
    min_outer_edge: float = 0.6  # m; shorter steps in the outline are smoothed away
    door_width: float = 0.9
    strip_bridge: float = 0.6    # m; interruptions of an unwalked strip shorter than this are bridged
    max_strip: float = 1.4       # m; widest unwalked strip that is still a wall candidate
    min_room_area: float = 1.0   # m^2
    split_radius: float = 0.55   # m; openings narrower than about twice this separate rooms
    min_seed_area: float = 0.3   # m^2
    raster: float = 0.05         # m; room segmentation resolution


@dataclass
class _Seg:
    vertical: bool
    c: float          # x for vertical walls, y for horizontal
    s0: float         # extent along the wall
    s1: float
    loss: float
    exterior: bool = False

    def wall(self) -> Wall:
        if self.vertical:
            return Wall(self.c, self.s0, self.c, self.s1, round(self.loss, 1), self.exterior)
        return Wall(self.s0, self.c, self.s1, self.c, round(self.loss, 1), self.exterior)


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r * r


def envelope(rti: RTIResult, p: ExtractParams) -> np.ndarray:
    """Floor area of the home: walked cells, lane gaps closed, holes filled, pushed out to the walls."""
    g = rti.grid
    vis = rti.visited
    rc = max(1, int(round(p.envelope_close / g.res)))
    pad = rc + 2
    v = np.pad(vis, pad)
    v = ndimage.binary_closing(v, structure=_disk(rc))[pad:-pad, pad:-pad]
    v = ndimage.binary_fill_holes(v)
    lab, n = ndimage.label(v)
    if n > 1:  # keep the main connected floor area
        sizes = ndimage.sum(v, lab, range(1, n + 1))
        v = lab == (1 + int(np.argmax(sizes)))
    rcl = int(round(p.clearance / g.res))
    if rcl > 0:
        v = ndimage.binary_dilation(v, structure=_disk(rcl))
    return v


def _strands(img: np.ndarray, vertical: bool, g: Grid, p: ExtractParams, weights: bool) -> list[_Seg]:
    """Cluster edge pieces along one orientation into wall segments.

    img is (rows, cols) with rows running *along* the wall direction for vertical
    walls (V: ny x nx+1) and across it for horizontal walls (H: ny+1 x nx); we
    transpose H so that axis 0 is always "along" and axis 1 "across".
    """
    a = img if vertical else img.T                   # (along, across)
    vals = np.nan_to_num(a, nan=0.0)
    mask = vals > (p.edge_min if weights else 0.5)
    k_across = max(1, int(round(p.merge_dist / g.res / 2)))
    k_along = max(1, int(round(p.gap_join / g.res / 2)))
    grown = ndimage.binary_dilation(mask, structure=np.ones((2 * k_along + 1, 2 * k_across + 1), bool))
    lab, n = ndimage.label(grown)
    origin_c = g.x0 if vertical else g.y0
    origin_s = g.y0 if vertical else g.x0
    out = []
    for comp in range(1, n + 1):
        sel = (lab == comp) & mask
        if not sel.any():
            continue
        rows, colsx = np.nonzero(sel)
        w = vals[rows, colsx] if weights else np.ones(len(rows))
        c = origin_c + (np.average(colsx, weights=w)) * g.res          # grid lines sit at integer index
        s0 = origin_s + rows.min() * g.res
        s1 = origin_s + (rows.max() + 1) * g.res
        # loss of one crossing = sum over parallel strands, typical value along the wall
        per_row = np.bincount(rows - rows.min(), weights=vals[rows, colsx]) if weights else np.zeros(1)
        loss = float(np.median(per_row[per_row > 0])) if weights and (per_row > 0).any() else 0.0
        out.append(_Seg(vertical, float(c), float(s0), float(s1), loss))
    return out


def interior_walls(rti: RTIResult, p: ExtractParams, loose: bool = False) -> list[_Seg]:
    """Wall candidates from the edge losses; `loose` keeps weak/short ones for the re-fit to judge."""
    segs = _strands(rti.v_loss(), True, rti.grid, p, True) + _strands(rti.h_loss(), False, rti.grid, p, True)
    min_len, min_loss = (p.min_len / 2, p.min_loss / 3) if loose else (p.min_len, p.min_loss)
    return [s for s in segs if (s.s1 - s.s0) >= min_len and s.loss >= min_loss]


def strip_candidates(env: np.ndarray, visited: np.ndarray, g: Grid, p: ExtractParams) -> list[_Seg]:
    """Wall hypotheses from geometry alone: long, narrow unwalked strips inside the home.

    You cannot walk through a wall, so every interior wall lies in such a strip
    (the converse is not true: gaps between walking lanes and furniture also
    leave strips).  These are only candidates; `refine` keeps the ones the radio
    data supports.
    """
    U = env & ~visited
    L = max(2, int(round(p.min_len / g.res)))
    bridge = max(1, int(round(p.strip_bridge / g.res / 2)))
    env_t = lambda vertical: env.T if vertical else env
    wmax = p.max_strip / g.res
    out = []
    for vertical in (False, True):
        u = U.T if vertical else U                     # rows = across, cols = along
        # tracking error makes the walk graze walls; bridge short interruptions along the strip
        u = ndimage.binary_closing(np.pad(u, 3), structure=np.ones((1, 2 * bridge + 1), bool))[3:-3, 3:-3] & env_t(vertical)
        long_ = ndimage.binary_opening(u, structure=np.ones((1, L), bool))
        lab, n = ndimage.label(long_)
        for k in range(1, n + 1):
            rows, cols = np.nonzero(lab == k)
            if rows.max() - rows.min() + 1 > wmax:
                continue                               # a wide unwalked area, not a strip
            across0, along0 = (g.x0, g.y0) if vertical else (g.y0, g.x0)
            c = across0 + (rows.mean() + 0.5) * g.res
            out.append(_Seg(vertical, float(c), float(along0 + cols.min() * g.res),
                            float(along0 + (cols.max() + 1) * g.res), 0.0))
    return out


def _dedupe(rf: list[_Seg], geo: list[_Seg], p: ExtractParams) -> list[_Seg]:
    """Drop geometric candidates that duplicate an RF candidate (collinear columns confuse the fit)."""
    keep = []
    for gs in geo:
        dup = False
        for r in rf:
            if r.vertical != gs.vertical or abs(r.c - gs.c) > p.merge_dist:
                continue
            ov = min(r.s1, gs.s1) - max(r.s0, gs.s0)
            if ov > 0.5 * (gs.s1 - gs.s0):
                dup = True
                break
        if not dup:
            keep.append(gs)
    return rf + keep


def _suppress_parallel(segs: list[_Seg], p: ExtractParams) -> list[_Seg]:
    """Of two parallel walls closer than `merge_dist` that mostly overlap, keep the stronger.

    Both hypotheses explain the same links, so the fit can keep two weaker copies of
    one wall.  (A real pair of walls this close, e.g. a closet, gets merged too.)
    """
    order = sorted(range(len(segs)), key=lambda k: -segs[k].loss * (segs[k].s1 - segs[k].s0))
    kept: list[_Seg] = []
    for k in order:
        s = segs[k]
        dup = any(o.vertical == s.vertical and abs(o.c - s.c) <= p.merge_dist
                  and min(o.s1, s.s1) - max(o.s0, s.s0) > 0.5 * (s.s1 - s.s0) for o in kept)
        if not dup:
            kept.append(s)
    return kept


def exterior_walls(env: np.ndarray, g: Grid, p: ExtractParams) -> list[_Seg]:
    """Outline of the floor area as a rectilinear polygon (radio links never cross the outer walls).

    The outline is traced, every edge snapped to horizontal/vertical, and the
    shortest edges removed one at a time (merging their neighbours) until no
    edge is shorter than `min_outer_edge`.  This removes the saw-tooth left by
    the ends of walking lanes while keeping real corners and alcoves.
    """
    cont = measure.find_contours(np.pad(env, 1).astype(float), 0.5)
    if not cont:
        return []
    c = measure.approximate_polygon(max(cont, key=len), tolerance=1.0)
    xy = np.column_stack([g.x0 + (c[:, 1] - 0.5) * g.res, g.y0 + (c[:, 0] - 0.5) * g.res])
    edges = []   # [vertical, coord, weight]
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-9:
            continue
        vertical = abs(d[1]) > abs(d[0])
        edges.append([vertical, float((a[0] + b[0]) / 2 if vertical else (a[1] + b[1]) / 2), L])

    def merge_runs(E):
        out = []
        for e in E:
            if out and out[-1][0] == e[0]:
                w = out[-1][2] + e[2]
                out[-1] = [e[0], (out[-1][1] * out[-1][2] + e[1] * e[2]) / w, w]
            else:
                out.append(list(e))
        if len(out) > 1 and out[0][0] == out[-1][0]:
            last = out.pop()
            w = out[0][2] + last[2]
            out[0] = [last[0], (out[0][1] * out[0][2] + last[1] * last[2]) / w, w]
        return out

    edges = merge_runs(edges)
    while len(edges) > 4:
        n = len(edges)
        lengths = [abs(edges[(k + 1) % n][1] - edges[k - 1][1]) for k in range(n)]
        k = int(np.argmin(lengths))
        if lengths[k] >= p.min_outer_edge:
            break
        del edges[k]
        edges = merge_runs(edges[k:] + edges[:k])
    if len(edges) < 4:
        return []
    n = len(edges)
    vert = []
    for k in range(n):
        e, f = edges[k], edges[(k + 1) % n]
        vert.append((f[1], e[1]) if not e[0] else (e[1], f[1]))
    segs = []
    for k in range(n):
        a, b = vert[k - 1], vert[k]
        vertical = edges[k][0]
        cc = a[0] if vertical else a[1]
        s0, s1 = sorted((a[1], b[1]) if vertical else (a[0], b[0]))
        segs.append(_Seg(vertical, float(cc), float(s0), float(s1), 0.0, True))
    return segs


def _join_collinear(segs: list[_Seg], traj: np.ndarray, p: ExtractParams) -> list[_Seg]:
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                if a.vertical != b.vertical or abs(a.c - b.c) > p.merge_dist / 2:
                    continue
                lo, hi = (a, b) if a.s0 <= b.s0 else (b, a)
                gap = hi.s0 - lo.s1
                if gap > p.door_join:
                    continue
                if gap > p.gap_join and not _crossed(lo.vertical, (lo.c + hi.c) / 2, lo.s1, hi.s0, traj):
                    continue
                la, lb = a.s1 - a.s0, b.s1 - b.s0
                segs[i] = _Seg(a.vertical, (a.c * la + b.c * lb) / (la + lb), min(a.s0, b.s0), max(a.s1, b.s1),
                               (a.loss * la + b.loss * lb) / (la + lb), a.exterior and b.exterior)
                del segs[j]
                changed = True
                break
            if changed:
                break
    return segs


def _crossed(vertical: bool, c: float, s0: float, s1: float, traj: np.ndarray) -> bool:
    return len(_crossings(vertical, c, s0, s1, traj)) > 0


def _crossings(vertical: bool, c: float, s0: float, s1: float, traj: np.ndarray) -> np.ndarray:
    """Positions along the wall line where the walk crosses it between s0 and s1."""
    if len(traj) < 2:
        return np.zeros(0)
    a = np.array([c, s0]) if vertical else np.array([s0, c])
    b = np.array([c, s1]) if vertical else np.array([s1, c])
    P, Q = traj[:-1], traj[1:]
    ok = np.linalg.norm(Q - P, axis=1) < 3.0
    hit = segments_intersect(P, Q, a, b) & ok
    if not hit.any():
        return np.zeros(0)
    k = 0 if vertical else 1   # coordinate across the wall
    P, Q = P[hit], Q[hit]
    t = (c - P[:, k]) / (Q[:, k] - P[:, k])
    along = P[:, 1 - k] + t * (Q[:, 1 - k] - P[:, 1 - k])
    return along


def _snap(segs: list[_Seg], p: ExtractParams) -> None:
    """Extend wall ends to the nearest perpendicular wall within `snap` metres (T and L junctions)."""
    for s in segs:
        if s.exterior:
            continue
        for end in (0, 1):
            pos = s.s0 if end == 0 else s.s1
            best = None
            for o in segs:
                if o.vertical == s.vertical:
                    continue
                if not (o.s0 - p.merge_dist / 2 <= s.c <= o.s1 + p.merge_dist / 2):
                    continue
                dist = (pos - o.c) if end == 0 else (o.c - pos)
                if -p.merge_dist / 2 <= dist <= p.snap and (best is None or abs(dist) < abs(best - pos)):
                    best = o.c
            if best is not None:   # extend to the junction, or trim a small overshoot
                if end == 0:
                    s.s0 = best
                else:
                    s.s1 = best


def _doors(segs: list[_Seg], traj: np.ndarray, p: ExtractParams) -> list[Door]:
    doors = []
    for wi, s in enumerate(segs):
        if s.exterior:   # the outer wall is drawn just outside the walk; crossings there are tracking noise
            continue
        along = np.sort(_crossings(s.vertical, s.c, s.s0, s.s1, traj))
        if len(along) == 0:
            continue
        # cluster crossings that are within a door width of each other
        groups = [[along[0]]]
        for v in along[1:]:
            if v - groups[-1][-1] < p.door_width:
                groups[-1].append(v)
            else:
                groups.append([v])
        for gp in groups:
            span = max(gp) - min(gp)
            width = max(p.door_width, span + 0.3)
            center = float(np.clip(np.mean(gp) - s.s0, width / 2, (s.s1 - s.s0) - width / 2))
            doors.append(Door(wi, round(center, 2), round(min(width, s.s1 - s.s0), 2)))
    return doors


def _rooms(env: np.ndarray, g: Grid, segs: list[_Seg], names: dict[str, tuple[float, float]],
           p: ExtractParams) -> list[Room]:
    f = max(1, int(round(g.res / p.raster)))
    r = g.res / f
    floor = np.kron(env, np.ones((f, f), bool))
    ny, nx = floor.shape
    blocked = np.zeros_like(floor)
    th = max(1, int(round(0.05 / r)))
    for s in segs:
        if s.vertical:
            i = int(round((s.c - g.x0) / r))
            j0, j1 = int(np.floor((s.s0 - g.y0) / r)), int(np.ceil((s.s1 - g.y0) / r))
            blocked[max(0, j0):max(0, j1), max(0, i - th):max(0, i + th)] = True
        else:
            j = int(round((s.c - g.y0) / r))
            i0, i1 = int(np.floor((s.s0 - g.x0) / r)), int(np.ceil((s.s1 - g.x0) / r))
            blocked[max(0, j - th):max(0, j + th), max(0, i0):max(0, i1)] = True
    free = floor & ~blocked
    # Walls found by radio can have gaps, so plain connected components would let rooms
    # leak into each other.  Morphological room segmentation instead: seeds are the
    # parts of the floor further than `split_radius` from any wall, grown back by
    # watershed, which splits the floor at every opening narrower than ~2*split_radius
    # (doors and wall gaps) while keeping wider areas whole.
    dist = ndimage.distance_transform_edt(free) * r
    seeds, ns = ndimage.label(dist > p.split_radius)
    if ns:
        sizes = ndimage.sum(np.ones_like(seeds), seeds, range(1, ns + 1))
        small = np.flatnonzero(sizes * r * r < p.min_seed_area) + 1
        seeds[np.isin(seeds, small)] = 0
    lab = segmentation.watershed(-dist, seeds, mask=free) if seeds.any() else ndimage.label(free)[0]
    # give the wall pixels back to the rooms on either side, so areas run to the wall centre line
    lab = segmentation.expand_labels(lab, distance=th + 1) * floor
    n = int(lab.max())
    rooms = []
    min_px = p.min_room_area / (r * r)
    for k in range(1, n + 1):
        m = lab == k
        area_px = int(m.sum())
        if area_px < min_px:
            continue
        cont = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
        if not cont:
            continue
        c = max(cont, key=len)
        c = measure.approximate_polygon(c, tolerance=1.5) - 1   # undo the pad
        poly = [(float(g.x0 + (x + 0.5) * r), float(g.y0 + (y + 0.5) * r)) for y, x in c]
        rooms.append((m, Room("", poly, round(area_px * r * r, 2))))
    # name rooms after the nodes in them (ESPresense nodes are usually named by room):
    # every node votes for the nearest room within 1 m
    rooms.sort(key=lambda t: -t[1].area_m2)
    votes: dict[int, list[str]] = {}
    for nid, (x, y) in sorted(names.items()):
        j, i = int((y - g.y0) / r), int((x - g.x0) / r)
        best, best_d = None, 1.0
        for idx, (m, _) in enumerate(rooms):
            dd = _dist_to_mask(m, j, i, int(1.0 / r)) * r
            if dd <= best_d:
                best, best_d = idx, dd
        if best is not None:
            votes.setdefault(best, []).append(nid)
    out = []
    for idx, (m, room) in enumerate(rooms):
        # several nodes in one room usually means a missed wall: keep all names so it shows
        room.name = "+".join(_dedupe_names(votes[idx])) if idx in votes else f"rom{idx + 1}"
        out.append(room)
    return out


def _dedupe_names(ids: list[str]) -> list[str]:
    """'stue', 'stue2' -> 'stue' (numbered nodes in the same room are not separate rooms)."""
    out = []
    for k in ids:
        base = k.rstrip("0123456789_-") or k
        if not any((o.rstrip("0123456789_-") or o) == base for o in out):
            out.append(k)
    return out


def _dist_to_mask(m: np.ndarray, j: int, i: int, rad: int) -> float:
    """Distance in pixels from (j, i) to the nearest True pixel of m, searched within `rad`."""
    j0, i0 = max(0, j - rad), max(0, i - rad)
    win = m[j0:j + rad + 1, i0:i + rad + 1]
    if not win.any():
        return np.inf
    yy, xx = np.nonzero(win)
    return float(np.min(np.hypot(yy + j0 - j, xx + i0 - i)))


def extract(rti: RTIResult, traj_xy: np.ndarray, nodes: dict[str, tuple[float, float]],
            p: ExtractParams | None = None, height: float = 2.4, agg: dict | None = None) -> FloorPlan:
    """Walls, doors and rooms from the tomography result.

    With `agg` (the aggregated link data) the interior wall candidates are
    re-fitted as whole walls (see `refine.py`); without it they are taken as is.
    """
    p = p or ExtractParams()
    g = rti.grid
    env = envelope(rti, p)
    inner = interior_walls(rti, p, loose=agg is not None)
    # an interior wall must lie on the floor area, and not in the band along the outer wall:
    # that band is never walked (it is the clearance) and links barely constrain it
    depth = ndimage.distance_transform_edt(np.pad(env, 1))[1:-1, 1:-1] * g.res
    inner = [s for s in inner if _on_floor(s, env, g) and _depth(s, depth, g) > p.clearance + g.res]
    inner = _join_collinear(inner, traj_xy, p)
    if agg is not None:
        geo = [s for s in strip_candidates(env, rti.visited, g, p) if _depth(s, depth, g) > p.clearance + g.res]
        inner = _dedupe(inner, geo, p)
        inner = refine(inner, agg, traj_xy, env, rti.visited, g, keep_db=p.min_loss)
        inner = [s for s in _join_collinear(inner, traj_xy, p) if s.s1 - s.s0 >= p.min_len]
        inner = _suppress_parallel(inner, p)
    outer = exterior_walls(env, g, p)
    segs = outer + inner
    _snap(segs, p)
    segs = [s for s in segs if s.exterior or s.s1 - s.s0 >= p.min_len / 2]   # snapping can shrink stubs
    doors = _doors(segs, traj_xy, p)
    rooms = _rooms(env, g, segs, nodes, p)
    return FloorPlan([s.wall() for s in segs], doors, rooms, height)


def _depth(s: _Seg, depth: np.ndarray, g: Grid) -> float:
    """Median distance (m) from points on the wall to the edge of the floor area."""
    t = np.linspace(s.s0, s.s1, max(2, int((s.s1 - s.s0) / g.res) + 1))
    x, y = (np.full_like(t, s.c), t) if s.vertical else (t, np.full_like(t, s.c))
    j = np.clip(((y - g.y0) / g.res).astype(int), 0, g.ny - 1)
    i = np.clip(((x - g.x0) / g.res).astype(int), 0, g.nx - 1)
    return float(np.median(depth[j, i]))


def _on_floor(s: _Seg, env: np.ndarray, g: Grid) -> bool:
    mid = (s.s0 + s.s1) / 2
    x, y = (s.c, mid) if s.vertical else (mid, s.c)
    j, i = int((y - g.y0) / g.res), int((x - g.x0) / g.res)
    return 0 <= j < g.ny and 0 <= i < g.nx and bool(env[max(0, j - 1):j + 2, max(0, i - 1):i + 2].any())
