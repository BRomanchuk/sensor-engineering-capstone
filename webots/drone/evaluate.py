"""Score + plot the drone capstone run.

Reads results/trajectory.csv (written by the Webots controller) and produces:
  * results/trajectory.png -- (left) x-y path, (right) altitude z over time,
    each showing ground truth, estimate, and raw GPS,
  * the 3D ATE (position RMSE) printed with the raw-GPS baseline.

Run OUTSIDE Webots:
    python evaluate.py
Needs numpy + matplotlib.
"""

import csv
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def find_csv():
    if len(sys.argv) > 1:
        candidates = [Path(sys.argv[1])]
    else:
        candidates = [
            HERE / "controllers" / "capstone_drone" / "results" / "trajectory.csv",
            HERE / "results" / "trajectory.csv",
            Path.cwd() / "results" / "trajectory.csv",
        ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "trajectory.csv not found. Run the world in Webots first, or pass the path:\n"
        "    python evaluate.py <path-to>/results/trajectory.csv\n"
        "Looked in:\n  " + "\n  ".join(str(c) for c in candidates))


def main():
    CSV = find_csv()
    print(f"Reading {CSV}")

    t = []
    gt, est = [], []          # (x,y,z)
    gps, gps_gt, gps_t = [], [], []
    vis, vis_t = [], []
    baro, baro_t = [], []
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            g = (float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"]))
            gt.append(g)
            est.append((float(row["est_x"]), float(row["est_y"]), float(row["est_z"])))
            if row["gps_x"] != "":
                gps.append((float(row["gps_x"]), float(row["gps_y"]), float(row["gps_z"])))
                gps_gt.append(g)
                gps_t.append(float(row["t"]))
            if row["vis_x"] != "":
                vis.append((float(row["vis_x"]), float(row["vis_y"]), float(row["vis_z"])))
                vis_t.append(float(row["t"]))
            if row["baro"] != "":
                baro.append(float(row["baro"]))
                baro_t.append(float(row["t"]))

    t = np.array(t)
    gt = np.array(gt); est = np.array(est)
    gps = np.array(gps); gps_gt = np.array(gps_gt); gps_t = np.array(gps_t)
    vis = np.array(vis); vis_t = np.array(vis_t)
    baro = np.array(baro); baro_t = np.array(baro_t)

    ate = math.sqrt(np.mean(np.linalg.norm(est - gt, axis=1) ** 2))
    gps_ate = (math.sqrt(np.mean(np.linalg.norm(gps - gps_gt, axis=1) ** 2))
               if len(gps) else float("nan"))

    fig, (axp, axz) = plt.subplots(1, 2, figsize=(13, 6))
    axp.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="Ground truth")
    axp.plot(est[:, 0], est[:, 1], "b-", lw=1.4, label=f"Estimate (3D ATE {ate:.3f} m)")
    if len(gps):
        axp.plot(gps[:, 0], gps[:, 1], "r.", ms=4, alpha=0.5, label="GPS fixes")
    if len(vis):
        axp.plot(vis[:, 0], vis[:, 1], "m.", ms=4, alpha=0.5, label="Camera fixes")
    axp.plot(gt[0, 0], gt[0, 1], "go", ms=9, label="start")
    axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)")
    axp.set_title("Drone: horizontal path"); axp.axis("equal")
    axp.grid(True, alpha=0.3); axp.legend()

    axz.plot(t, gt[:, 2], "k-", lw=2, label="Ground truth z")
    axz.plot(t, est[:, 2], "b-", lw=1.4, label="Estimate z")
    if len(gps):
        axz.plot(gps_t, gps[:, 2], "r.", ms=4, alpha=0.5, label="GPS z")
    if len(baro):
        axz.plot(baro_t, baro, "g.", ms=4, alpha=0.5, label="Barometer z")
    axz.set_xlabel("t (s)"); axz.set_ylabel("z (m)")
    axz.set_title("Drone: altitude over time")
    axz.grid(True, alpha=0.3); axz.legend()

    fig.tight_layout()
    out = CSV.parent / "trajectory.png"
    fig.savefig(out, dpi=130)
    print(f"3D ATE (position RMSE): {ate:.3f} m   (raw-GPS baseline ~ {gps_ate:.3f} m)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
