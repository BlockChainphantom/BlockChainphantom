"""Thin-wall radio tomography.

Homes are made of thin walls, so instead of the classic RTI pixel image
(Wilson & Patwari, IEEE TMC 2010) the unknowns here are the *edges* of a square
grid: every vertical and horizontal cell edge may hold a piece of wall with a
loss ``x_e >= 0`` in dB per crossing.  Each averaged link measurement gives one
linear equation

    rssi = o_dev - 10 n log10(d) - sum_e  cross(link, e) * x_e

``o_dev`` is one offset per fixed node for walker links (TX power + antenna gain
of both ends) and ``o_a + o_b + k`` for node <-> node links, which is an exact
re-parametrisation of per-device gains.  The path-loss exponent ``n`` is
estimated jointly with a weak prior, and so is the walker's body shadowing
``b * body_factor`` (see `data.body_factor`).

Geometry from the walk constrains the problem hard:
  * an edge the walker crossed is open (no wall, or a door),
  * an edge whose two neighbouring cells were both walked through is open.
Remaining edges are solved with FISTA under non-negativity, an L1 penalty
(walls are sparse) and a continuity term between collinear neighbouring edges
(walls are long and straight).  The grid is aligned with the house first (see
``dominant_angle``), so this also encodes the Manhattan prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as splinalg

from .data import WALKER


@dataclass
class Grid:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int

    @classmethod
    def around(cls, pts: np.ndarray, res: float, pad: float) -> "Grid":
        lo = pts.min(0) - pad
        hi = pts.max(0) + pad
        n = np.ceil((hi - lo) / res).astype(int)
        return cls(float(lo[0]), float(lo[1]), res, int(n[0]), int(n[1]))

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def n_v(self) -> int:
        """Vertical edges: lines x = x0 + i*res (i = 0..nx), one per row j."""
        return self.ny * (self.nx + 1)

    @property
    def n_h(self) -> int:
        """Horizontal edges: lines y = y0 + j*res (j = 0..ny), one per column i."""
        return (self.ny + 1) * self.nx

    def index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(int)
        return np.clip(iy, 0, self.ny - 1), np.clip(ix, 0, self.nx - 1)

    def v_index(self, j, i):
        return np.asarray(j) * (self.nx + 1) + np.asarray(i)

    def h_index(self, j, i):
        return self.n_v + np.asarray(j) * self.nx + np.asarray(i)


def _crossings_1d(lo_c, hi_c, origin, res, n_lines):
    """For each segment spanning coordinate [lo_c, hi_c], the grid-line indices strictly inside."""
    first = np.floor((lo_c - origin) / res).astype(int) + 1
    last = np.ceil((hi_c - origin) / res).astype(int) - 1
    first = np.clip(first, 0, n_lines)
    last = np.clip(last, -1, n_lines - 1)
    cnt = np.maximum(0, last - first + 1)
    rows = np.repeat(np.arange(len(cnt)), cnt)
    starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
    lines = first[rows] + (np.arange(cnt.sum()) - starts)
    return rows, lines


def edge_crossings(grid: Grid, pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, edge) pairs for every grid edge crossed by the segments pa[r] -> pb[r]."""
    out_r, out_e = [], []
    # vertical edges: x = x0 + i*res
    r, i = _crossings_1d(np.minimum(pa[:, 0], pb[:, 0]), np.maximum(pa[:, 0], pb[:, 0]), grid.x0, grid.res, grid.nx + 1)
    if len(r):
        xl = grid.x0 + i * grid.res
        t = (xl - pa[r, 0]) / (pb[r, 0] - pa[r, 0])
        y = pa[r, 1] + t * (pb[r, 1] - pa[r, 1])
        j = np.floor((y - grid.y0) / grid.res).astype(int)
        ok = (j >= 0) & (j < grid.ny)
        out_r.append(r[ok]); out_e.append(grid.v_index(j[ok], i[ok]))
    # horizontal edges: y = y0 + j*res
    r, j = _crossings_1d(np.minimum(pa[:, 1], pb[:, 1]), np.maximum(pa[:, 1], pb[:, 1]), grid.y0, grid.res, grid.ny + 1)
    if len(r):
        yl = grid.y0 + j * grid.res
        t = (yl - pa[r, 1]) / (pb[r, 1] - pa[r, 1])
        x = pa[r, 0] + t * (pb[r, 0] - pa[r, 0])
        i = np.floor((x - grid.x0) / grid.res).astype(int)
        ok = (i >= 0) & (i < grid.nx)
        out_r.append(r[ok]); out_e.append(grid.h_index(j[ok], i[ok]))
    if not out_r:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(out_r), np.concatenate(out_e)


def crossing_matrix(grid: Grid, pa: np.ndarray, pb: np.ndarray) -> sparse.csr_matrix:
    r, e = edge_crossings(grid, pa, pb)
    m = sparse.csr_matrix((np.ones(len(r)), (r, e)), shape=(len(pa), grid.n_v + grid.n_h))
    m.data[:] = 1.0  # a straight link crosses an edge at most once
    return m


def visited_cells(grid: Grid, traj_xy: np.ndarray, body_radius: float) -> np.ndarray:
    """Cells the walker's body passed through (path densified, then dilated)."""
    m = np.zeros(grid.shape, bool)
    if len(traj_xy) == 0:
        return m
    pts = [traj_xy]
    for p, q in zip(traj_xy[:-1], traj_xy[1:]):
        L = np.linalg.norm(q - p)
        if grid.res / 2 < L < 3.0:   # densify normal steps, skip tracking jumps
            t = np.linspace(0, 1, int(np.ceil(L / grid.res * 3)) + 1)[1:-1, None]
            pts.append(p + t * (q - p))
    iy, ix = grid.index(np.vstack(pts))
    m[iy, ix] = True
    r = int(round(body_radius / grid.res))
    if r > 0:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = ndimage.binary_dilation(m, structure=(xx ** 2 + yy ** 2) <= r * r)
    return m


def open_edges(grid: Grid, traj_xy: np.ndarray, visited: np.ndarray, door_halfwidth: int = 1) -> np.ndarray:
    """Edges that cannot hold a wall: crossed by the walk, or between two visited cells."""
    is_open = np.zeros(grid.n_v + grid.n_h, bool)
    if len(traj_xy) > 1:
        ok = np.linalg.norm(np.diff(traj_xy, axis=0), axis=1) < 3.0
        if ok.any():
            _, e = edge_crossings(grid, traj_xy[:-1][ok], traj_xy[1:][ok])
            is_open[e] = True
    V = is_open[:grid.n_v].reshape(grid.ny, grid.nx + 1)
    H = is_open[grid.n_v:].reshape(grid.ny + 1, grid.nx)
    if door_halfwidth > 0:
        # a body is wider than one edge: open the neighbours along the same line too
        V = ndimage.binary_dilation(V, structure=np.ones((2 * door_halfwidth + 1, 1), bool))
        H = ndimage.binary_dilation(H, structure=np.ones((1, 2 * door_halfwidth + 1), bool))
    V[:, 1:-1] |= visited[:, :-1] & visited[:, 1:]
    H[1:-1, :] |= visited[:-1, :] & visited[1:, :]
    return np.concatenate([V.ravel(), H.ravel()])


@dataclass
class RTIResult:
    grid: Grid
    loss: np.ndarray          # (n_v + n_h,) dB per crossing, NaN where open/unobserved
    hits: np.ndarray          # (n_v + n_h,) link rows crossing each edge
    visited: np.ndarray       # (ny, nx)
    n_exp: float
    body_db: float           # estimated body shadowing (dB) when walking directly away from a node
    offsets: dict[str, float]
    rms_residual: float
    rms_residual_no_walls: float

    def v_loss(self) -> np.ndarray:
        return self.loss[:self.grid.n_v].reshape(self.grid.ny, self.grid.nx + 1)

    def h_loss(self) -> np.ndarray:
        return self.loss[self.grid.n_v:].reshape(self.grid.ny + 1, self.grid.nx)


def solve(agg: dict, traj_xy: np.ndarray, grid: Grid, *, body_radius: float = 0.15,
          l1: float = 0.05, smooth: float = 0.5, n_prior: float = 2.0, n_prior_w: float = 0.05,
          min_hits: int = 3, iters: int = 2000) -> RTIResult:
    """Joint estimate of wall losses on grid edges, node offsets and path-loss exponent.

    Penalties are scaled by the summed row weights so their meaning does not
    change with survey length.
    """
    a, b, pa, pb, rssi = agg["a"], agg["b"], agg["pa"], agg["pb"], agg["rssi"]
    m = len(rssi)
    w = np.sqrt(np.minimum(agg["n"], 4.0))         # fast fading averages out, shadowing does not
    d = np.maximum(np.linalg.norm(pb - pa, axis=1), 0.3)

    C = crossing_matrix(grid, pa, pb)
    hits = np.asarray(C.sum(0)).ravel()
    visited = visited_cells(grid, traj_xy, body_radius)
    unknown = (~open_edges(grid, traj_xy, visited)) & (hits >= min_hits)
    cols = np.flatnonzero(unknown)
    Cx = C[:, cols]

    nodes = sorted((set(a.tolist()) | set(b.tolist())) - {WALKER})
    ni = {k: i for i, k in enumerate(nodes)}
    walker_row = (a == WALKER) | (b == WALKER)
    has_nn = bool((~walker_row).any())
    rr, cc, vv = [], [], []
    for r in range(m):
        if walker_row[r]:
            rr.append(r); cc.append(ni[b[r] if a[r] == WALKER else a[r]]); vv.append(1.0)
        else:
            rr += [r, r, r]; cc += [ni[a[r]], ni[b[r]], len(nodes)]; vv += [1.0, 1.0, 1.0]
    O = sparse.csr_matrix((vv, (rr, cc)), shape=(m, len(nodes) + 1))
    if not has_nn:
        O = O[:, :len(nodes)]
    logd = sparse.csr_matrix((-10 * np.log10(d))[:, None])
    body = sparse.csr_matrix(-np.asarray(agg.get("body", np.zeros(m)), float)[:, None])
    Wr = sparse.diags(w)
    A = (Wr @ sparse.hstack([-Cx, O, body, logd])).tocsr()
    y = w * rssi
    scale = float(np.sum(w ** 2))

    # column scaling (preconditioning)
    cn = np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel())
    cn[cn == 0] = 1.0
    S = 1.0 / cn
    As = (A @ sparse.diags(S)).tocsr()
    nx_ = len(cols)
    i_n = As.shape[1] - 1
    i_b = i_n - 1
    D = (_collinear_diff(grid, cols) @ sparse.diags(S[:nx_])).tocsr()
    lam_s = smooth * scale / max(1, nx_)
    lam_n = n_prior_w * scale

    def grad(u):
        g = As.T @ (As @ u - y)
        g[:nx_] += lam_s * (D.T @ (D @ u[:nx_]))
        g[i_n] += lam_n * S[i_n] * (S[i_n] * u[i_n] - n_prior)
        return g

    def hess_vec(v):
        h = As.T @ (As @ v)
        h[:nx_] += lam_s * (D.T @ (D @ v[:nx_]))
        h[i_n] += lam_n * S[i_n] ** 2 * v[i_n]
        return h

    step = 1.0 / (_power_iter(hess_vec, As.shape[1]) * 1.05)
    thr = step * l1 * np.sqrt(scale) * S[:nx_]

    # warm start: offsets and n from a wall-free least-squares fit
    B = (Wr @ sparse.hstack([O, body, logd])).tocsr()
    sol0 = splinalg.lsqr(B, y, atol=1e-10, btol=1e-10)[0]
    res0 = (B @ sol0 - y) / w
    u = np.zeros(As.shape[1])
    u[nx_:] = sol0 / S[nx_:]
    z, tk = u.copy(), 1.0
    for _ in range(iters):
        un = z - step * grad(z)
        un[:nx_] = np.maximum(0.0, un[:nx_] - thr)
        un[i_n] = np.clip(un[i_n] * S[i_n], 1.4, 4.5) / S[i_n]
        un[i_b] = np.clip(un[i_b] * S[i_b], 0.0, 15.0) / S[i_b]
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = un + ((tk - 1) / tn) * (un - u)
        u, tk = un, tn

    sol = u * S
    loss = np.full(grid.n_v + grid.n_h, np.nan)
    loss[cols] = sol[:nx_]
    res = (A @ sol - y) / w
    return RTIResult(
        grid=grid, loss=loss, hits=hits, visited=visited, n_exp=float(sol[i_n]), body_db=float(sol[i_b]),
        offsets={k: float(sol[nx_ + i]) for k, i in ni.items()},
        rms_residual=float(np.sqrt(np.mean(res ** 2))),
        rms_residual_no_walls=float(np.sqrt(np.mean(res0 ** 2))),
    )


def _collinear_diff(grid: Grid, cols: np.ndarray) -> sparse.csr_matrix:
    """Difference operator between each unknown edge and its collinear neighbour."""
    pos = -np.ones(grid.n_v + grid.n_h, int)
    pos[cols] = np.arange(len(cols))
    nbr = np.full(len(cols), -1)
    isv = cols < grid.n_v
    j, i = np.divmod(cols[isv], grid.nx + 1)            # vertical edge -> same line, next row
    nv = np.full(len(j), -1)
    ok = j + 1 < grid.ny
    nv[ok] = pos[grid.v_index(j[ok] + 1, i[ok])]
    nbr[isv] = nv
    j, i = np.divmod(cols[~isv] - grid.n_v, grid.nx)     # horizontal edge -> same line, next column
    nh = np.full(len(j), -1)
    ok = i + 1 < grid.nx
    nh[ok] = pos[grid.h_index(j[ok], i[ok] + 1)]
    nbr[~isv] = nh
    src = np.flatnonzero(nbr >= 0)
    k = len(src)
    rows = np.concatenate([np.arange(k), np.arange(k)])
    return sparse.csr_matrix((np.concatenate([np.ones(k), -np.ones(k)]),
                              (rows, np.concatenate([src, nbr[src]]))), shape=(k, len(cols)))


def _power_iter(op, n: int, iters: int = 80) -> float:
    v = np.random.default_rng(0).normal(size=n)
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(iters):
        hv = op(v)
        lam = float(np.linalg.norm(hv))
        if lam == 0:
            return 1.0
        v = hv / lam
    return lam


def dominant_angle(traj_xy: np.ndarray, chord: float = 1.5) -> float:
    """House orientation in [-45°, 45°), from walking directions (people walk along walls).

    Headings are taken over ~`chord` metres of path so tracking wobble averages
    out, folded modulo 90° and the *peak* of their distribution is returned
    (diagonal walks between doors do not shift a peak the way they shift a mean).
    """
    if len(traj_xy) < 3:
        return 0.0
    st = np.linalg.norm(np.diff(traj_xy, axis=0), axis=1)
    st[st > 3.0] = 0.0                      # tracking jumps
    s = np.concatenate([[0.0], np.cumsum(st)])
    j = np.searchsorted(s, s + chord)
    ok = j < len(s)
    i, j = np.flatnonzero(ok), j[ok]
    if len(i) == 0:
        return 0.0
    d = traj_xy[j] - traj_xy[i]
    straight = np.linalg.norm(d, axis=1) > 0.8 * chord          # skip turns
    if not straight.any():
        return 0.0
    ang = np.mod(np.arctan2(d[straight, 1], d[straight, 0]), np.pi / 2)
    grid = np.deg2rad(np.arange(0, 90, 0.1))
    diff = np.angle(np.exp(4j * (grid[:, None] - ang[None, :]))) / 4    # wrapped to +-45°
    score = np.exp(-0.5 * (diff / np.deg2rad(2.0)) ** 2).sum(axis=1)
    best = grid[int(np.argmax(score))]
    return float(best - np.pi / 2 if best >= np.pi / 4 else best)
