"""Capstone DRONE (quadrotor) controller (Webots).

What this does, out of the box:
  * flies the drone along a horizontal loop while its altitude rises and falls
    (kinematically, via the Supervisor -- the drone is physics-free so we place
    it exactly),
  * synthesises the noisy sensor readings from the known true state,
  * hands them to your estimator (student_estimator.StudentEstimator),
  * logs true vs estimated trajectory to results/trajectory.csv and prints the
    3D ATE at the end.

Your job (the capstone): open student_estimator.py and replace the default
pass-through with your own fusion filter (KF / EKF / UKF). This world adds a
third dimension: you estimate (x, y, z, yaw). The BAROMETER gives a smooth,
frequent altitude, where the GPS z is noisiest -- fusing them is the point.

Coordinate frame: x forward, y left, z up. Yaw = rotation about z.
Run from Webots: open worlds/capstone_drone.wbt and press Play.
"""

import os
import csv
import math
import random
import numpy as np

from controller import Supervisor

from student_estimator import StudentEstimator

# ---------------------------------------------------------------- configuration
RUN_SECONDS = 60.0
V_HORIZ = 0.72             # horizontal speed (m/s)
OMEGA = 0.12              # turn rate (rad/s) -> loop radius ~ V/OMEGA = 6 m
BASE_ALT = 4.0            # mean altitude (m)
ALT_AMP = 1.5            # altitude swing (m)
ALT_W = 0.20            # altitude angular frequency (rad/s)
START = (0.0, -6.0)

GPS_PERIOD_S = 1.0
GPS_NOISE_XY = 0.40      # metres, horizontal GPS noise
GPS_NOISE_Z = 1.20      # metres, VERTICAL GPS noise (much worse -- realistic)

BARO_NOISE_STD = 0.15   # metres, barometer/altimeter noise (smooth, frequent)
BARO_PERIOD_S = 0.1           # seconds, barometer update period (smooth, frequent)

GYRO_BIAS = 0.000
GYRO_NOISE_STD = 0.01
SPEED_NOISE_STD = 0.03

CAMERA_SHIFT_STD = 0.02  # metres, camera pixel shift noise (smooth, frequent)
CAMERA_ROT_STD = 0.01      # radians, camera rotation noise (smooth, frequent)
CAMERA_SCALE_STD = 0.0001
CAMERA_PERIOD_S = 0.1           # seconds, camera update period (smooth, frequent)

MAG_NOISE_STD = 0.01      # radians, magnetometer noise (smooth, frequent)
MAG_PERIOD_S = 0.5             # seconds, magnetometer update period (smooth, frequent)

RESULTS_DIR = "results"

random.seed(0)

def psi_to_mag(psi: float) -> np.ndarray:
    """
    Convert heading (yaw) in radians to magnetometer measurement.

    Args:
        psi: heading (yaw) in radians
    """
    mx = math.cos(psi)
    my = math.sin(psi)
    mz = 0.0
    return np.array([mx, my, mz])


def main():
    robot = Supervisor()
    dt_ms = int(robot.getBasicTimeStep())
    dt = dt_ms / 1000.0

    self_node = robot.getSelf()
    trans = self_node.getField("translation")
    rot = self_node.getField("rotation")

    estimator = StudentEstimator()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = open(os.path.join(RESULTS_DIR, "trajectory.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["t", "gt_x", "gt_y", "gt_z", "gt_yaw",
                     "est_x", "est_y", "est_z", "est_yaw",
                     "gps_x", "gps_y", "gps_z"])

    x, y = START
    psi = 0.0
    t = 0.0

    since_gps = GPS_PERIOD_S
    since_baro = BARO_PERIOD_S
    since_camera = CAMERA_PERIOD_S
    since_mag = MAG_PERIOD_S

    sq_err_sum = 0.0
    n = 0

    last_vis_px = x
    last_vis_py = y
    last_vis_pz = BASE_ALT
    last_vis_heading = psi

    while robot.step(dt_ms) != -1:
        t += dt
        since_gps += dt
        since_baro += dt
        since_mag += dt
        since_camera += dt

        # --- advance the TRUE state ------------------------------------------
        psi += OMEGA * dt
        x += V_HORIZ * math.cos(psi) * dt
        y += V_HORIZ * math.sin(psi) * dt
        z = BASE_ALT + ALT_AMP * math.sin(ALT_W * t)

        vx = V_HORIZ * math.cos(psi)
        vy = V_HORIZ * math.sin(psi)
        vz = ALT_AMP * ALT_W * math.cos(ALT_W * t)

        ax = -V_HORIZ * OMEGA * math.sin(psi)
        ay = V_HORIZ * OMEGA * math.cos(psi)
        az = -ALT_AMP * ALT_W * ALT_W * math.sin(ALT_W * t)

        trans.setSFVec3f([x, y, z])
        rot.setSFRotation([0, 0, 1, psi])

        # --- synthesise the sensors ------------------------------------------
        gyro_z = OMEGA + GYRO_BIAS + random.gauss(0, GYRO_NOISE_STD)
        speed = V_HORIZ + random.gauss(0, SPEED_NOISE_STD)          # horizontal

        baro = None
        if since_baro >= BARO_PERIOD_S:
            since_baro = 0.0
            baro = z + random.gauss(0, BARO_NOISE_STD)                  # smooth altitude, every tick

        imu_meas = np.array([ax, ay, az + 9.81, 0.0, 0.0, gyro_z])  # ax, ay, az, gx, gy, gz
        gps_xyz = None
        if since_gps >= GPS_PERIOD_S:
            since_gps = 0.0
            gps_xyz = (x + random.gauss(0, GPS_NOISE_XY),
                       y + random.gauss(0, GPS_NOISE_XY),
                       z + random.gauss(0, GPS_NOISE_Z))
            
        mag = None
        # if since_mag >= MAG_PERIOD_S:
        #     since_mag = 0.0
        #     mag = psi_to_mag(psi + random.gauss(0, MAG_NOISE_STD))

        camera = None
        if since_camera >= CAMERA_PERIOD_S:
            since_camera = 0.0
            # --- camera sees the world from the drone's perspective -----------
            # 2D transformation matrix in drone frame:
            scale = (last_vis_pz / z)
            d_heading = (psi - last_vis_heading)
            relative_shift_world = np.array([x - last_vis_px, y - last_vis_py])
            relative_shift_body = np.array([
                relative_shift_world[0] * np.cos(-last_vis_heading) - relative_shift_world[1] * np.sin(-last_vis_heading),
                relative_shift_world[0] * np.sin(-last_vis_heading) + relative_shift_world[1] * np.cos(-last_vis_heading)
            ]) * scale

            d_heading += random.gauss(0, CAMERA_ROT_STD)
            scale += random.gauss(0, CAMERA_SCALE_STD)
            R = np.array([[np.cos(d_heading), -np.sin(d_heading)],
                            [np.sin(d_heading),  np.cos(d_heading)]])
            
            camera = np.eye(2, 3)
            camera[:2, :2] = R * scale
            camera[0, 2] = relative_shift_body[0] + random.gauss(0, CAMERA_SHIFT_STD)

            last_vis_px = x
            last_vis_py = y
            last_vis_pz = z
            last_vis_heading = psi

        sensors = {
            "t": t,
            "dt": dt,
            "imu": imu_meas,     # yaw rate (rad/s), biased
            "camera": camera,       # horizontal speed (m/s)
            "baro": baro,         # altitude (m): smooth + frequent, low noise
            "gps": gps_xyz,       # absolute (x, y, z) with noise, or None between updates
            "mag": mag  # heading (rad), smooth + frequent, low noise
        }

        est_x, est_y, est_z, est_yaw = estimator.estimate(sensors)

        # --- score against ground truth (3D) ---------------------------------
        writer.writerow([f"{t:.3f}", x, y, z, psi, est_x, est_y, est_z, est_yaw,
                         "" if gps_xyz is None else gps_xyz[0],
                         "" if gps_xyz is None else gps_xyz[1],
                         "" if gps_xyz is None else gps_xyz[2]])
        sq_err_sum += (est_x - x) ** 2 + (est_y - y) ** 2 + (est_z - z) ** 2
        n += 1

        if t >= RUN_SECONDS:
            break

    log.close()
    ate = math.sqrt(sq_err_sum / max(n, 1))
    print(f"[capstone-drone] done: {n} steps, 3D ATE (position RMSE) = {ate:.3f} m")
    print(f"[capstone-drone] trajectory saved to {os.path.join(RESULTS_DIR, 'trajectory.csv')}")
    print("[capstone-drone] plot it with:  python evaluate.py")


if __name__ == "__main__":
    main()
