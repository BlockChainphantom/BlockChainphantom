import json

import numpy as np
import pytest

from rfhome.data import WALKER, Survey, aggregate, body_factor
from rfhome.export import _triangulate, mesh, write_gltf, write_obj
from rfhome.geometry import Door, FloorPlan, Wall, segments_intersect
from rfhome.ingest import build_survey, parse_message
from rfhome.tomography import Grid, dominant_angle, edge_crossings


def test_segments_intersect():
    p = np.array([[0.0, 0.0], [0.0, 2.0]])
    q = np.array([[2.0, 2.0], [2.0, 3.0]])
    hit = segments_intersect(p, q, np.array([0.0, 2.0]), np.array([2.0, 0.0]))
    assert hit.tolist() == [True, False]


def test_solid_pieces_cut_doors():
    plan = FloorPlan([Wall(0, 0, 4, 0, 5)], [Door(0, 2.0, 1.0)])
    pieces = [(w.x0, w.x1) for _, w in plan.solid_pieces()]
    assert pieces == [(0, 1.5), (2.5, 4)]


def test_edge_crossings_counts_lines():
    g = Grid(0.0, 0.0, 1.0, 5, 5)
    r, e = edge_crossings(g, np.array([[0.5, 0.5]]), np.array([[3.5, 0.5]]))
    assert len(e) == 3                      # crosses x = 1, 2, 3
    assert all(k < g.n_v for k in e)        # all vertical edges
    r, e = edge_crossings(g, np.array([[0.5, 0.5]]), np.array([[2.5, 2.5 + 1e-9]]))
    assert len(e) == 4                      # two vertical + two horizontal


@pytest.mark.parametrize("deg", [0.0, 12.0, -30.0, 44.0])
def test_dominant_angle(deg):
    # lawn-mower walk in a rotated frame
    pts = []
    for k in range(8):
        y = (0.0, 5.0) if k % 2 == 0 else (5.0, 0.0)
        pts += [(k * 0.8, y[0]), (k * 0.8, y[1])]
    pts = np.array(pts)
    dense = np.vstack([np.linspace(a, b, 20, endpoint=False) for a, b in zip(pts[:-1], pts[1:])])
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    est = dominant_angle(dense @ R.T)
    assert abs(np.rad2deg(est) - deg) < 0.5


def _tiny_survey():
    traj = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]], float)
    t = np.array([0.5, 0.6, 1.5, 2.5])
    return Survey(
        t=t, a=np.array([WALKER, "n1", WALKER, "n1"]), b=np.array(["n1", WALKER, "n1", "n2"]),
        rssi=np.array([-50.0, -52.0, -60.0, -70.0]),
        pa=np.array([[0.5, 0], [4, 0], [1.5, 0], [4, 0]], float),
        pb=np.array([[4, 0], [0.55, 0], [4, 0], [4, 4]], float),
        traj=traj, nodes={"n1": (4.0, 0.0), "n2": (4.0, 4.0)})


def test_aggregate_merges_reciprocal_links():
    agg = aggregate(_tiny_survey(), cell=0.25)
    # first two rows are the same link (walker<->n1) from the same 25 cm cell
    assert len(agg["rssi"]) == 3
    k = int(np.argmax(agg["n"]))
    assert agg["n"][k] == 2 and agg["rssi"][k] == -51.0
    walker_end = agg["pa"][k] if agg["a"][k] == WALKER else agg["pb"][k]
    assert np.allclose(walker_end, [0.525, 0.0])


def test_body_factor_walking_away():
    s = _tiny_survey()
    f = body_factor(s)
    # walker moves +x at 1 m/s, node n1 is ahead (+x) -> not shadowed
    assert f[0] == pytest.approx(0.0)
    assert f[3] == 0.0                      # node<->node row


def test_save_load_roundtrip(tmp_path):
    s = _tiny_survey()
    s.save(tmp_path)
    s2 = Survey.load(tmp_path)
    assert s2.a.tolist() == s.a.tolist() and np.allclose(s2.pa, s.pa) and s2.nodes == s.nodes


def test_parse_espresense():
    assert parse_message("espresense/devices/phone:abc/stue", b'{"rssi": -71.5, "distance": 3.1}') == ("phone:abc", "stue", -71.5)
    assert parse_message("espresense/devices/node:kjokken/stue", '{"rssi": -80}') == ("kjokken", "stue", -80.0)
    assert parse_message("espresense/rooms/stue/telemetry", "{}") is None
    assert parse_message("espresense/devices/x/stue", "not json") is None


def test_build_survey_interpolates_walker():
    raw = [(10.5, "me", "a", -60.0), (10.5, "other", "a", -40.0), (11.0, "b", "a", -70.0), (99.0, "me", "a", -60.0)]
    traj = np.array([[10.0, 0.0, 0.0], [11.0, 2.0, 0.0]])
    s = build_survey(raw, "me", traj, {"a": (5.0, 0.0), "b": (5.0, 5.0)})
    assert len(s) == 2                       # stranger device and far-off timestamp dropped
    assert np.allclose(s.pa[0], [1.0, 0.0]) and s.b[0] == "a"
    assert s.a[1] == "b" and np.allclose(s.pa[1], [5.0, 5.0])


def test_triangulate_l_shape_area():
    P = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], float)
    v, f = _triangulate(P)
    tri = v[f][:, :, :2]
    u, w = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum()
    assert area == pytest.approx(12.0)


def test_mesh_exports(tmp_path):
    plan = FloorPlan([Wall(0, 0, 4, 0), Wall(4, 0, 4, 3)], [Door(0, 2.0, 0.9)])
    v, f = mesh(plan)["walls"]
    assert len(f) == 12 * 4                  # two wall pieces + lintel + second wall
    assert f.max() < len(v)
    write_obj(plan, tmp_path / "m.obj")
    write_gltf(plan, tmp_path / "m.gltf")
    doc = json.loads((tmp_path / "m.gltf").read_text())
    assert doc["asset"]["version"] == "2.0" and doc["buffers"][0]["byteLength"] > 0
