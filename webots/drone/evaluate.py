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

from scipy.stats import chi2

from controllers.capstone_drone.scenarios import USE_GPS, USE_BARO, USE_MAG, USE_CAMERA


NIS_THRESHOLDS = {
   "gps": chi2.ppf(0.99, df=3),
    "baro": chi2.ppf(0.99, df=1),
    "camera": chi2.ppf(0.99, df=3),
    "mag": chi2.ppf(0.99, df=1),
}


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
    est_yaw = []
    vis_yaw = []
    gt_yaw = []
    gps, gps_gt, gps_t = [], [], []
    vis, vis_t = [], []
    baro, baro_t = [], []
    mag, mag_t = [], []
    gps_nis, baro_nis, mag_nis, vis_nis = [], [], [], []
    gps_fault, baro_fault, mag_fault, vis_fault = [], [], [], []
    gps_health_t, baro_health_t, mag_health_t, vis_health_t = [], [], [], []
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            g = (float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"]))
            gt_yaw.append(float(row["gt_yaw"]))
            est_yaw.append(float(row["est_yaw"]))
            gt.append(g)
            est.append((float(row["est_x"]), float(row["est_y"]), float(row["est_z"])))
            if row["gps_x"] != "":
                gps.append((float(row["gps_x"]), float(row["gps_y"]), float(row["gps_z"])))
                gps_gt.append(g)
                gps_t.append(float(row["t"]))
            if row["vis_x"] != "":
                vis.append((float(row["vis_x"]), float(row["vis_y"]), float(row["vis_z"])))
                vis_yaw.append(float(row["vis_yaw"]))
                vis_t.append(float(row["t"]))
            if row["baro"] != "":
                baro.append(float(row["baro"]))
                baro_t.append(float(row["t"]))
            if row["mag"] != "":
                mag.append(float(row["mag"]))
                mag_t.append(float(row["t"]))
            if row["gps_nis"] != "":
                gps_nis.append(float(row["gps_nis"]))
                gps_fault.append(row["gps_fault"] == "True")
                gps_health_t.append(float(row["t"]))
            if row["baro_nis"] != "":
                baro_nis.append(float(row["baro_nis"]))
                baro_fault.append(row["baro_fault"] == "True")
                baro_health_t.append(float(row["t"]))
            if row["mag_nis"] != "":
                mag_nis.append(float(row["mag_nis"]))
                mag_fault.append(row["mag_fault"] == "True")
                mag_health_t.append(float(row["t"]))
            if row["vis_nis"] != "":
                vis_nis.append(float(row["vis_nis"]))
                vis_fault.append(row["vis_fault"] == "True")
                vis_health_t.append(float(row["t"]))
            
            

    t = np.array(t)
    gt = np.array(gt); est = np.array(est)
    gps = np.array(gps); gps_gt = np.array(gps_gt); gps_t = np.array(gps_t)
    vis = np.array(vis); vis_t = np.array(vis_t)
    baro = np.array(baro); baro_t = np.array(baro_t)

    ate = math.sqrt(np.mean(np.linalg.norm(est - gt, axis=1) ** 2))
    gps_ate = (math.sqrt(np.mean(np.linalg.norm(gps - gps_gt, axis=1) ** 2))
               if len(gps) else float("nan"))
    

    # 4 subplots: x-y path, z over time, yaws, NIS for each sensor
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # fig, (axp, axz) = plt.subplots(2, 2, figsize=(13, 6))
    # Horizontal path graph
    axs[0][0].plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="Ground truth")
    axs[0][0].plot(est[:, 0], est[:, 1], "b-", lw=1.4, label=f"Estimate (3D ATE {ate:.3f} m)")
    if len(gps):
        axs[0][0].plot(gps[:, 0], gps[:, 1], "r.", ms=4, alpha=0.5, label="GPS fixes")
    if len(vis):
        axs[0][0].plot(vis[:, 0], vis[:, 1], "m.", ms=4, alpha=0.1, label="Camera fixes")
    axs[0][0].plot(gt[0, 0], gt[0, 1], "go", ms=9, label="start")
    axs[0][0].set_xlabel("x (m)"); axs[0][0].set_ylabel("y (m)")
    axs[0][0].set_title("Drone: horizontal path"); axs[0][0].axis("equal")
    axs[0][0].grid(True, alpha=0.3); axs[0][0].legend()

    axs[0][1].plot(t, gt[:, 2], "k-", lw=2, label="Ground truth z")
    axs[0][1].plot(t, est[:, 2], "b-", lw=1.4, label="Estimate z")
    if len(gps):
        axs[0][1].plot(gps_t, gps[:, 2], "r.", ms=4, alpha=0.5, label="GPS z")
    if len(baro):
        axs[0][1].plot(baro_t, baro, "g.", ms=4, alpha=0.5, label="Barometer z")
    axs[0][1].set_xlabel("t (s)"); axs[0][1].set_ylabel("z (m)")
    axs[0][1].set_title("Drone: altitude over time")
    axs[0][1].grid(True, alpha=0.3); axs[0][1].legend()

    # yaw plots
    # axs[1][0].plot(t, gt[:, 3], "k-", lw=2, label="Ground truth yaw")
    axs[1][0].plot(t, np.rad2deg(est_yaw)%360, "b-", lw=1.4, label="Estimate yaw")
    if len(mag):
        axs[1][0].plot(mag_t, np.rad2deg(mag)%360, "g.", ms=4, alpha=0.5, label="Magnetometer yaw")
    if len(vis):
        axs[1][0].plot(vis_t, np.rad2deg(vis_yaw)%360, "m.", ms=4, alpha=0.5, label="Camera yaw")
    
    
    axs[1][0].set_xlabel("t (s)"); axs[1][0].set_ylabel("yaw (deg)")
    axs[1][0].set_title("Drone: yaw over time")
    axs[1][0].grid(True, alpha=0.3); axs[1][0].legend()

    # NIS plots
    if len(gps_nis):
        axs[1][1].plot(gps_health_t, gps_nis, "r.", ms=4, alpha=0.5, label="GPS NIS")
    if len(baro_nis):
        axs[1][1].plot(baro_health_t, baro_nis, "g.", ms=4, alpha=0.5, label="Barometer NIS")
    if len(mag_nis):
        axs[1][1].plot(mag_health_t, mag_nis, "b.", ms=4, alpha=0.5, label="Magnetometer NIS")
    if len(vis_nis):
        axs[1][1].plot(vis_health_t, vis_nis, "m.", ms=4, alpha=0.5, label="Camera NIS")
    
    axs[1][1].axhline(NIS_THRESHOLDS["gps"], color="r", ls="--", lw=1, alpha=0.9, label="GPS & Cam NIS thres.")
    axs[1][1].axhline(NIS_THRESHOLDS["baro"], color="g", ls="--", lw=1, alpha=0.9, label="Baro & Mag NIS thres.")
    # detected faults
    if len(gps_fault) > 0:
        axs[1][1].scatter(np.array(gps_health_t)[np.array(gps_fault)], 20*np.ones_like(gps_nis)[np.array(gps_fault)], color="r", s=40, marker="x", label="GPS fault")
    if len(baro_fault) > 0:
        axs[1][1].scatter(np.array(baro_health_t)[np.array(baro_fault)], 19*np.ones_like(baro_nis)[np.array(baro_fault)], color="g", s=40, marker="x", label="Barometer fault")
    if len(mag_fault) > 0:
        axs[1][1].scatter(np.array(mag_health_t)[np.array(mag_fault)], 18*np.ones_like(mag_nis)[np.array(mag_fault)], color="b", s=40, marker="x", label="Magnetometer fault")
    if len(vis_fault) > 0:
        axs[1][1].scatter(np.array(vis_health_t)[np.array(vis_fault)], 17*np.ones_like(vis_nis)[np.array(vis_fault)], color="m", s=40, marker="x", label="Camera fault")
    # limit y axis to show NIS values and thresholds clearly
    axs[1][1].set_ylim(0, 20)
    axs[1][1].set_xlabel("t (s)"); axs[1][1].set_ylabel("NIS")
    axs[1][1].set_title("Drone: NIS over time")
    axs[1][1].grid(True, alpha=0.3); 
    axs[1][1].legend(loc="upper left", fontsize=6)

    

    fig.tight_layout()
    out = CSV.parent / f"trajectory{'-gps' if USE_GPS else ""}{'-baro' if USE_BARO else ""}{'-mag' if USE_MAG else ""}{'-vis' if USE_CAMERA else ""}.png"
    fig.savefig(out, dpi=130)
    print(f"3D ATE (position RMSE): {ate:.3f} m   (raw-GPS baseline ~ {gps_ate:.3f} m)")
    print(f"Terminal Error (3D distance to goal): {np.linalg.norm(est[-1] - gt[-1]):.3f} m")
    print(f"Total mission distance: {np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)):.3f} m")
    print(f"Saved {out}")



if __name__ == "__main__":
    main()
