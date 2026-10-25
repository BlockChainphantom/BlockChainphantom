"""Exports: 2D plot (PNG), 3D mesh (OBJ, glTF) and a self-contained 3D viewer (HTML)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .geometry import FloorPlan

DOOR_HEIGHT = 2.05
WALL_THICKNESS = 0.1


# --------------------------------------------------------------------------- 2D
def plot(plan: FloorPlan, path: str | Path, *, truth: FloorPlan | None = None, traj: np.ndarray | None = None,
         nodes: dict | None = None, rti=None, rti_angle: float = 0.0, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(10, 8))
    if rti is not None:
        g = rti.grid
        c, s = np.cos(rti_angle), np.sin(rti_angle)
        R = np.array([[c, -s], [s, c]])
        segs, vals = [], []
        V, H = rti.v_loss(), rti.h_loss()
        for j, i in zip(*np.nonzero(np.nan_to_num(V) > 0.3)):
            x = g.x0 + i * g.res
            segs.append(np.array([(x, g.y0 + j * g.res), (x, g.y0 + (j + 1) * g.res)]) @ R.T); vals.append(V[j, i])
        for j, i in zip(*np.nonzero(np.nan_to_num(H) > 0.3)):
            y = g.y0 + j * g.res
            segs.append(np.array([(g.x0 + i * g.res, y), (g.x0 + (i + 1) * g.res, y)]) @ R.T); vals.append(H[j, i])
        if segs:
            lc = LineCollection(segs, array=np.array(vals), cmap="magma_r", linewidths=2, alpha=0.6)
            lc.set_clim(0, max(3.0, float(np.percentile(vals, 98))))
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="RF wall loss per grid edge (dB)", shrink=0.7)
    for k, r in enumerate(plan.rooms):
        P = np.asarray(r.polygon)
        ax.fill(P[:, 0], P[:, 1], color=plt.cm.Pastel1(k % 9), alpha=0.5, lw=0)
        cx, cy = P.mean(0)
        ax.text(cx, cy, f"{r.name}\n{r.area_m2:.1f} m²", ha="center", va="center", fontsize=9)
    if truth is not None:
        for _, w in truth.solid_pieces():
            ax.plot([w.x0, w.x1], [w.y0, w.y1], color="tab:green", lw=7, alpha=0.3, solid_capstyle="butt")
    for _, w in plan.solid_pieces():
        ax.plot([w.x0, w.x1], [w.y0, w.y1], color="k", lw=3 if w.exterior else 2, solid_capstyle="butt")
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=0.5, alpha=0.6)
    for nid, (x, y) in (nodes or {}).items():
        ax.plot(x, y, "^", color="tab:red", ms=9)
        ax.annotate(nid, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color="tab:red")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title or "Reconstructed floor plan" + ("  (green = ground truth)" if truth else ""))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- 3D mesh
def _box(p0, p1, z0, z1, t):
    """Vertices/triangles of a wall slab from p0 to p1 (2D), thickness t, height z0..z1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6 or z1 <= z0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    n = np.array([-d[1], d[0]]) / L * t / 2
    q = [p0 - n, p1 - n, p1 + n, p0 + n]
    v = np.array([[*xy, z] for z in (z0, z1) for xy in q])
    f = np.array([
        [0, 2, 1], [0, 3, 2],          # bottom
        [4, 5, 6], [4, 6, 7],          # top
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return v, f


def mesh(plan: FloorPlan) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named meshes ('walls', 'floor') as (vertices, faces); z is up."""
    H = plan.height
    parts = []
    for i, w in enumerate(plan.walls):
        a, b = np.array([w.x0, w.y0]), np.array([w.x1, w.y1])
        L = w.length
        if L == 0:
            continue
        u = (b - a) / L
        gaps = sorted((max(0, d.center - d.width / 2), min(L, d.center + d.width / 2)) for d in plan.doors if d.wall == i)
        s = 0.0
        for g0, g1 in gaps + [(L, L)]:
            if g0 > s:
                parts.append(_box(a + u * s, a + u * g0, 0, H, WALL_THICKNESS))
            if g1 > g0:   # lintel above the door
                parts.append(_box(a + u * g0, a + u * g1, min(DOOR_HEIGHT, H), H, WALL_THICKNESS))
            s = max(s, g1)
    floor = []
    for r in plan.rooms:
        P = np.asarray(r.polygon, float)
        if len(P) >= 3:
            floor.append(_triangulate(P))
    return {"walls": _merge(parts), "floor": _merge(floor)}


def _merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        if len(v) == 0:
            continue
        vs.append(v); fs.append(f + off); off += len(v)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(vs), np.vstack(fs)


def _triangulate(P: np.ndarray):
    """Ear clipping for a simple polygon (room outlines are small, so O(n^2) is fine)."""
    P = P[:-1] if np.allclose(P[0], P[-1]) else P
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    idx = list(range(len(P))) if area > 0 else list(range(len(P)))[::-1]
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue
            if any(_in_tri(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append([i0, i1, i2])
            idx.pop(k)
            break
        else:
            break   # degenerate outline; stop rather than loop
    if len(idx) == 3:
        tris.append(idx)
    v = np.column_stack([P, np.zeros(len(P))])
    return v, np.array(tris, int).reshape(-1, 3)


def _in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def write_obj(plan: FloorPlan, path: str | Path) -> None:
    lines = ["# rfhome floor plan, metres, z up"]
    off = 0
    for name, (v, f) in mesh(plan).items():
        lines.append(f"o {name}")
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v]
        lines += [f"f {a + 1 + off} {b + 1 + off} {c + 1 + off}" for a, b, c in f]
        off += len(v)
    Path(path).write_text("\n".join(lines) + "\n")


def write_gltf(plan: FloorPlan, path: str | Path) -> None:
    """Minimal glTF 2.0 with an embedded buffer (y up, as glTF requires)."""
    colors = {"walls": [0.92, 0.92, 0.9, 1.0], "floor": [0.75, 0.62, 0.45, 1.0]}
    buf = bytearray()
    views, accessors, meshes, nodes, materials = [], [], [], [], []
    for name, (v, f) in mesh(plan).items():
        if len(f) == 0:
            continue
        vy = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype(np.float32)   # z-up -> y-up
        fi = f.astype(np.uint32)
        for arr, target in ((vy, 34962), (fi, 34963)):
            while len(buf) % 4:
                buf.append(0)
            views.append({"buffer": 0, "byteOffset": len(buf), "byteLength": arr.nbytes, "target": target})
            buf += arr.tobytes()
        accessors.append({"bufferView": len(views) - 2, "componentType": 5126, "count": len(vy), "type": "VEC3",
                          "min": vy.min(0).tolist(), "max": vy.max(0).tolist()})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125, "count": fi.size, "type": "SCALAR"})
        materials.append({"name": name, "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": colors[name], "metallicFactor": 0, "roughnessFactor": 0.9}})
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": len(accessors) - 2},
                                                     "indices": len(accessors) - 1, "material": len(materials) - 1}]})
        nodes.append({"mesh": len(meshes) - 1, "name": name})
    doc = {
        "asset": {"version": "2.0", "generator": "rfhome"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
        "materials": materials, "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(buf)).decode()}],
    }
    Path(path).write_text(json.dumps(doc))


VIEWER = """<!doctype html>
<html lang="no"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RF Home Model</title>
<style>
:root{--bg:#f4f3ef;--fg:#222;--panel:#ffffffdd}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e8e8e8;--panel:#1f2226dd}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e8e8e8;--panel:#1f2226dd}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;overflow:hidden}
#info{position:absolute;top:12px;left:12px;max-width:min(320px,calc(100% - 48px));background:var(--panel);padding:10px 12px;border-radius:8px}
#info h1{font-size:15px;margin:0 0 6px} #info ul{margin:4px 0 0;padding-left:18px}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body><div id="info"><h1>Rekonstruert hjem</h1><div id="sum"></div><ul id="rooms"></ul></div>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const PLAN = __PLAN__;
const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const cam = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 500);
const ctl = new OrbitControls(cam, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666655, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(5, 12, 8); scene.add(sun);
const H = PLAN.height, T = 0.1, DH = Math.min(2.05, H);
const wallMat = new THREE.MeshStandardMaterial({color:0xe9e7e1, roughness:0.9});
const extMat = new THREE.MeshStandardMaterial({color:0xcfcac0, roughness:0.9});
function slab(ax, ay, bx, by, z0, z1, mat){
  const L = Math.hypot(bx-ax, by-ay); if (L < 1e-3 || z1 <= z0) return;
  const m = new THREE.Mesh(new THREE.BoxGeometry(L, z1-z0, T), mat);
  m.position.set((ax+bx)/2, (z0+z1)/2, -(ay+by)/2); m.rotation.y = Math.atan2(by-ay, bx-ax); scene.add(m);
}
PLAN.walls.forEach((w, i) => {
  const L = Math.hypot(w.x1-w.x0, w.y1-w.y0); if (!L) return;
  const ux = (w.x1-w.x0)/L, uy = (w.y1-w.y0)/L, mat = w.exterior ? extMat : wallMat;
  const gaps = PLAN.doors.filter(d => d.wall === i).map(d => [Math.max(0, d.center-d.width/2), Math.min(L, d.center+d.width/2)]).sort((a,b)=>a[0]-b[0]);
  let s = 0;
  for (const [g0, g1] of [...gaps, [L, L]]) {
    if (g0 > s) slab(w.x0+ux*s, w.y0+uy*s, w.x0+ux*g0, w.y0+uy*g0, 0, H, mat);
    if (g1 > g0) slab(w.x0+ux*g0, w.y0+uy*g0, w.x0+ux*g1, w.y0+uy*g1, DH, H, mat);
    s = Math.max(s, g1);
  }
});
const hues = [0.08, 0.33, 0.58, 0.75, 0.95, 0.15, 0.45];
const roomsEl = document.getElementById("rooms");
PLAN.rooms.forEach((r, k) => {
  const shape = new THREE.Shape(r.polygon.map(([x, y]) => new THREE.Vector2(x, y)));
  const m = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({color: new THREE.Color().setHSL(hues[k % hues.length], 0.35, 0.7), side: THREE.DoubleSide}));
  m.rotation.x = -Math.PI/2; scene.add(m);
  const li = document.createElement("li"); li.textContent = `${r.name}: ${r.area_m2.toFixed(1)} m²`; roomsEl.appendChild(li);
});
(PLAN.nodes || []).forEach(n => {
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.12), new THREE.MeshStandardMaterial({color:0xd04040}));
  s.position.set(n.x, 1.0, -n.y); scene.add(s);
});
const total = PLAN.rooms.reduce((a, r) => a + r.area_m2, 0);
document.getElementById("sum").textContent = `${PLAN.rooms.length} rom, ${total.toFixed(1)} m², ${PLAN.walls.length} vegger, ${PLAN.doors.length} døråpninger`;
const box = new THREE.Box3().setFromObject(scene), c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3()).length();
ctl.target.copy(c); cam.position.set(c.x + sz*0.4, sz*0.8, c.z + sz*0.7); ctl.update();
addEventListener("resize", () => { cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
renderer.setAnimationLoop(() => { ctl.update(); renderer.render(scene, cam); });
</script></body></html>
"""


def write_viewer(plan: FloorPlan, path: str | Path, nodes: dict | None = None) -> None:
    data = json.loads(plan.to_json())
    data["nodes"] = [{"id": k, "x": float(v[0]), "y": float(v[1])} for k, v in (nodes or {}).items()]
    Path(path).write_text(VIEWER.replace("__PLAN__", json.dumps(data)))


def write_all(plan: FloorPlan, folder: str | Path, nodes: dict | None = None) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {"plan": folder / "floorplan.json", "obj": folder / "home.obj", "gltf": folder / "home.gltf",
           "html": folder / "home.html"}
    out["plan"].write_text(plan.to_json())
    write_obj(plan, out["obj"])
    write_gltf(plan, out["gltf"])
    write_viewer(plan, out["html"], nodes)
    return out
