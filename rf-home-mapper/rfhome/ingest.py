"""Collecting real data from ESPresense nodes over MQTT.

ESPresense (https://github.com/ESPresense/ESPresense) runs on ESP32 boards and
publishes every BLE advert it hears to::

    espresense/devices/<device_id>/<node_id>   {"rssi": -67, "distance": 2.3, ...}

Node-to-node packets use ``device_id = "node:<transmitting node>"`` (enable
"Node to node" / "Transmit" in the node settings so the nodes also measure the
walls between each other).

Two ways to get walker positions:

* ``survey``: stop-and-go.  You stand on known points (``waypoints.csv``:
  ``name,x,y``) and press Enter; packets are recorded while you stand still and
  the walk between points is assumed to be a straight line, so pick points so
  that straight lines go through doorways.
* ``record`` + ``join``: walk continuously while something else records your
  position (``t,x,y`` in unix seconds, e.g. ``tools/walker.html``), then join
  the two logs by timestamp.  Both clocks must agree (NTP on both).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .data import WALKER, Survey

TOPIC = "espresense/devices/#"


def parse_message(topic: str, payload: bytes | str) -> tuple[str, str, float] | None:
    """(transmitter, receiver_node, rssi) from one ESPresense MQTT message, or None."""
    parts = topic.split("/")
    if len(parts) != 4 or parts[0] != "espresense" or parts[1] != "devices":
        return None
    device, node = parts[2], parts[3]
    try:
        msg = json.loads(payload)
        rssi = float(msg["rssi"])
    except (ValueError, KeyError, TypeError):
        return None
    if device.startswith("node:"):
        device = device[len("node:"):]
    return device, node, rssi


class Recorder:
    """Subscribe to ESPresense and keep (t, transmitter, node, rssi) rows."""

    def __init__(self, host: str, port: int = 1883, username: str | None = None, password: str | None = None):
        import paho.mqtt.client as mqtt   # optional dependency

        self.rows: list[tuple[float, str, str, float]] = []
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = lambda c, *_: c.subscribe(TOPIC)
        self.client.on_message = self._on_message
        self.client.connect(host, port)

    def _on_message(self, _client, _userdata, msg):
        parsed = parse_message(msg.topic, msg.payload)
        if parsed:
            self.rows.append((time.time(), *parsed))

    def start(self):
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


def write_raw(rows, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "tx", "rx", "rssi"])
        for t, tx, rx, rssi in rows:
            w.writerow([f"{t:.3f}", tx, rx, f"{rssi:.1f}"])


def read_raw(path: str | Path) -> list[tuple[float, str, str, float]]:
    with open(path, newline="") as f:
        return [(float(r["t"]), r["tx"], r["rx"], float(r["rssi"])) for r in csv.DictReader(f)]


def build_survey(raw, walker_id: str, traj: np.ndarray, nodes: dict[str, tuple[float, float]],
                 max_gap: float = 1.0) -> Survey:
    """Combine raw packets with walker trajectory (t,x,y) and fixed node positions.

    Walker packets are placed at the walker position interpolated at their
    timestamp; packets more than `max_gap` seconds from any trajectory sample are
    dropped.  Node-to-node packets need both nodes in `nodes`.  Packets from other
    BLE devices (neighbours' phones, watches...) are ignored.
    """
    t_all, a_all, b_all, r_all, pa, pb = [], [], [], [], [], []
    tt = traj[:, 0] if len(traj) else np.zeros(0)
    for t, tx, rx, rssi in raw:
        if rx not in nodes:
            continue
        if tx == walker_id:
            if len(tt) == 0:
                continue
            k = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
            if min(abs(tt[k] - t), abs(tt[k - 1] - t)) > max_gap:
                continue
            pos = (float(np.interp(t, tt, traj[:, 1])), float(np.interp(t, tt, traj[:, 2])))
            t_all.append(t); a_all.append(WALKER); b_all.append(rx); r_all.append(rssi)
            pa.append(pos); pb.append(nodes[rx])
        elif tx in nodes and tx != rx:
            t_all.append(t); a_all.append(tx); b_all.append(rx); r_all.append(rssi)
            pa.append(nodes[tx]); pb.append(nodes[rx])
    if not t_all:
        raise ValueError("no usable packets: check --walker id, node names in nodes.json, and timestamps")
    return Survey(np.array(t_all), np.array(a_all), np.array(b_all), np.array(r_all, float),
                  np.array(pa, float).reshape(-1, 2), np.array(pb, float).reshape(-1, 2),
                  traj, dict(nodes))


def waypoint_survey(rec: Recorder, waypoints: list[tuple[str, float, float]], dwell: float,
                    ask=input, say=print) -> tuple[list, np.ndarray]:
    """Interactive stop-and-go survey.  Returns (raw rows, trajectory).

    While standing, turn slowly once around so the body does not always shadow
    the same nodes.  The trajectory holds one sample per waypoint at the start
    and end of the dwell, so the straight line between consecutive waypoints is
    treated as the walked path.
    """
    traj = []
    rec.start()
    try:
        for name, x, y in waypoints:
            ask(f"Go to '{name}' ({x:.2f}, {y:.2f}) and press Enter ")
            t0 = time.time()
            n0 = len(rec.rows)
            time.sleep(dwell)
            t1 = time.time()
            traj += [(t0, x, y), (t1, x, y)]
            say(f"  {len(rec.rows) - n0} packets")
    finally:
        rec.stop()
    return list(rec.rows), np.array(traj, float).reshape(-1, 3)
