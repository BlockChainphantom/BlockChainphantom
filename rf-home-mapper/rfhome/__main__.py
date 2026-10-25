"""Command line: python -m rfhome <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _nodes(path) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in json.loads(Path(path).read_text()).items()}


def cmd_simulate(a):
    from .simulate import example_home, simulate_survey
    plan, rooms, links, nodes = example_home()
    s = simulate_survey(plan, rooms, links, nodes, seed=a.seed)
    s.save(a.out)
    (Path(a.out) / "truth.json").write_text(plan.to_json())
    print(f"wrote {len(s)} packets, {len(s.traj)} trajectory samples and truth.json to {a.out}")


def cmd_reconstruct(a):
    from .data import Survey
    from .evaluate import compare
    from .export import plot, write_all
    from .extract import ExtractParams
    from .geometry import FloorPlan
    from .reconstruct import Params, reconstruct

    s = Survey.load(a.survey)
    ep = ExtractParams(clearance=a.clearance, min_loss=a.min_wall_db)
    p = Params(res=a.res, height=a.height, extract=ep,
               angle=None if a.angle_deg is None else np.deg2rad(a.angle_deg))
    t0 = time.time()
    r = reconstruct(s, p)
    out = Path(a.out)
    files = write_all(r.plan, out, s.nodes)
    truth = None
    truth_path = Path(a.truth) if a.truth else Path(a.survey) / "truth.json"
    if truth_path.exists():
        truth = FloorPlan.from_json(truth_path.read_text())
    plot(r.plan, out / "floorplan.png", truth=truth, traj=s.traj[:, 1:3], nodes=s.nodes, rti=r.rti, rti_angle=r.angle)
    report = {
        "seconds": round(time.time() - t0, 2),
        "house_angle_deg": round(float(np.rad2deg(r.angle)), 2),
        "path_loss_exponent": round(r.rti.n_exp, 2),
        "body_shadowing_db": round(r.rti.body_db, 1),
        "rms_residual_db": round(r.rti.rms_residual, 2),
        "rms_residual_without_walls_db": round(r.rti.rms_residual_no_walls, 2),
        "walls": len(r.plan.walls), "doors": len(r.plan.doors),
        "rooms": {rm.name: rm.area_m2 for rm in r.plan.rooms},
    }
    if truth is not None:
        report["accuracy_vs_truth"] = compare(r.plan, truth)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\n".join(f"{k:>5}: {v}" for k, v in {**files, "png": out / "floorplan.png"}.items()))


def cmd_evaluate(a):
    from .evaluate import compare
    from .geometry import FloorPlan
    m = compare(FloorPlan.from_json(Path(a.pred).read_text()), FloorPlan.from_json(Path(a.truth).read_text()))
    print(json.dumps(m, indent=2))


def cmd_record(a):
    from .ingest import Recorder, write_raw
    rec = Recorder(a.host, a.port, a.user, a.password)
    rec.start()
    print(f"recording espresense/devices/# from {a.host} — Ctrl+C to stop")
    try:
        end = time.time() + a.seconds if a.seconds else float("inf")
        while time.time() < end:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    rec.stop()
    write_raw(rec.rows, a.out)
    print(f"wrote {len(rec.rows)} packets to {a.out}")


def cmd_survey(a):
    import csv
    from .ingest import Recorder, build_survey, waypoint_survey, write_raw
    with open(a.waypoints, newline="") as f:
        wps = [(r["name"], float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
    nodes = _nodes(a.nodes)
    rows, traj = waypoint_survey(Recorder(a.host, a.port, a.user, a.password), wps, a.dwell)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_raw(rows, out / "raw.csv")
    s = build_survey(rows, a.walker, traj, nodes)
    s.save(out)
    print(f"saved {len(s)} usable packets to {out}")


def cmd_join(a):
    from .ingest import build_survey, read_raw
    traj = np.loadtxt(a.traj, delimiter=",", skiprows=1, usecols=(0, 1, 2), ndmin=2)
    traj = traj[np.argsort(traj[:, 0])]
    s = build_survey(read_raw(a.raw), a.walker, traj, _nodes(a.nodes))
    s.save(a.out)
    print(f"saved {len(s)} usable packets to {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rfhome", description="Floor plan / 3D model of a home from radio signal strength.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="generate a synthetic survey with known ground truth")
    p.add_argument("--out", default="sim_survey")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("reconstruct", help="survey folder -> floor plan, 3D model, viewer")
    p.add_argument("survey")
    p.add_argument("--out", default="model")
    p.add_argument("--res", type=float, default=0.2, help="grid resolution in metres")
    p.add_argument("--height", type=float, default=2.4, help="ceiling height in metres")
    p.add_argument("--angle-deg", type=float, default=None, help="house orientation; estimated if omitted")
    p.add_argument("--clearance", type=float, default=0.35, help="how close to the outer walls you walked (m)")
    p.add_argument("--min-wall-db", type=float, default=2.0, help="weakest wall to keep (dB per crossing)")
    p.add_argument("--truth", default=None, help="ground-truth floorplan.json to score against")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("evaluate", help="score a floorplan.json against a ground truth")
    p.add_argument("pred"); p.add_argument("truth")
    p.set_defaults(func=cmd_evaluate)

    for name, fn, hlp in (("record", cmd_record, "log ESPresense packets to raw.csv"),
                          ("survey", cmd_survey, "interactive stop-and-go survey with ESPresense")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--host", required=True); p.add_argument("--port", type=int, default=1883)
        p.add_argument("--user"); p.add_argument("--password")
        if name == "record":
            p.add_argument("--out", default="raw.csv")
            p.add_argument("--seconds", type=float, default=0)
        else:
            p.add_argument("--waypoints", required=True, help="CSV name,x,y")
            p.add_argument("--nodes", required=True, help="nodes.json {id: [x, y]}")
            p.add_argument("--walker", required=True, help="ESPresense device id of the phone/beacon you carry")
            p.add_argument("--dwell", type=float, default=15.0, help="seconds per waypoint")
            p.add_argument("--out", default="survey")
        p.set_defaults(func=fn)

    p = sub.add_parser("join", help="raw.csv + trajectory.csv (t,x,y) -> survey folder")
    p.add_argument("--raw", required=True); p.add_argument("--traj", required=True)
    p.add_argument("--nodes", required=True); p.add_argument("--walker", required=True)
    p.add_argument("--out", default="survey")
    p.set_defaults(func=cmd_join)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
