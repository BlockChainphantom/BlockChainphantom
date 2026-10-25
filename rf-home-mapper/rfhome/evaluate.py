"""Compare a reconstructed plan with a ground-truth plan."""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon as fill_polygon

from .geometry import FloorPlan, point_segment_distance


def _wall_points(plan: FloorPlan, step: float, interior_only: bool = False) -> np.ndarray:
    pts = [w.points(step) for _, w in plan.solid_pieces() if not (interior_only and w.exterior)]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _dist_to_walls(pts: np.ndarray, plan: FloorPlan) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros(0)
    d = np.full(len(pts), np.inf)
    for _, w in plan.solid_pieces():
        d = np.minimum(d, point_segment_distance(pts, np.array([w.x0, w.y0]), np.array([w.x1, w.y1])))
    return d


def _footprint(plan: FloorPlan, x0, y0, nx, ny, res) -> np.ndarray:
    m = np.zeros((ny, nx), bool)
    for r in plan.rooms:
        P = np.asarray(r.polygon)
        rr, cc = fill_polygon((P[:, 1] - y0) / res, (P[:, 0] - x0) / res, m.shape)
        m[rr, cc] = True
    return m


def compare(pred: FloorPlan, truth: FloorPlan, tol: float = 0.4, step: float = 0.05) -> dict:
    """Wall precision/recall within `tol` metres, wall position error, footprint IoU, room areas."""
    out = {}
    for label, interior in (("all", False), ("interior", True)):
        tp = _wall_points(truth, step, interior)
        pp = _wall_points(pred, step, interior)
        d_t = _dist_to_walls(tp, pred)   # truth -> prediction (recall)
        d_p = _dist_to_walls(pp, truth)  # prediction -> truth (precision)
        rec = float(np.mean(d_t <= tol)) if len(d_t) else float("nan")
        prec = float(np.mean(d_p <= tol)) if len(d_p) else float("nan")
        out[f"{label}_recall"] = rec
        out[f"{label}_precision"] = prec
        out[f"{label}_f1"] = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
        out[f"{label}_median_err_m"] = float(np.median(d_p[d_p <= tol])) if (d_p <= tol).any() else float("nan")
    allp = np.vstack([np.asarray(r.polygon) for r in truth.rooms + pred.rooms]) if truth.rooms and pred.rooms else None
    if allp is not None:
        res = 0.05
        x0, y0 = allp.min(0) - 0.5
        nx, ny = (np.ceil((allp.max(0) + 0.5 - [x0, y0]) / res)).astype(int)
        a = _footprint(truth, x0, y0, nx, ny, res)
        b = _footprint(pred, x0, y0, nx, ny, res)
        out["footprint_iou"] = float((a & b).sum() / max(1, (a | b).sum()))
    out["rooms_true"] = len(truth.rooms)
    out["rooms_found"] = len(pred.rooms)
    t_area = {r.name: r.area_m2 for r in truth.rooms}
    out["room_area_err"] = {r.name: round(r.area_m2 - t_area[r.name], 2) for r in pred.rooms if r.name in t_area}
    return out
