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
import config

# ---------------------------------------------------------------- configuration
RUN_SECONDS = 60.0
V_HORIZ = 0.72             # horizontal speed (m/s)
OMEGA = 0.12              # turn rate (rad/s) -> loop radius ~ V/OMEGA = 6 m
BASE_ALT = 4.0            # mean altitude (m)
ALT_AMP = 1.5            # altitude swing (m)
ALT_W = 0.20            # altitude angular frequency (rad/s)
START = (0.0, -6.0)


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

def rotate_vector(v: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 2D vector by a given angle.

    Args:
        v: 2D vector (shape: (2,))
        angle: rotation angle in radians

    Returns:
        Rotated 2D vector (shape: (2,))
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return R @ v


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
                     "gps_x", "gps_y", "gps_z", 
                     "vis_x", "vis_y", "vis_z", "vis_yaw", 
                     "baro", "mag",
                     "gps_nis", "baro_nis", "mag_nis", "vis_nis",
                     "gps_fault", "baro_fault", "mag_fault", "vis_fault"])

    x, y = START
    psi = 0.0
    t = 0.0

    since_gps = config.GPS_PERIOD_S
    since_baro = config.BARO_PERIOD_S
    since_camera = config.CAMERA_PERIOD_S
    since_mag = config.MAG_PERIOD_S

    sq_err_sum = 0.0
    n = 0

    last_vis_px = x
    last_vis_py = y
    last_vis_pz = BASE_ALT
    last_vis_heading = psi + np.pi/2

    vis_x = []
    vis_y = []

    while robot.step(dt_ms) != -1:
        t += dt
        since_gps += dt
        since_baro += dt
        since_mag += dt
        since_camera += dt

        # --- advance the TRUE state ------------------------------------------
        psi += OMEGA * dt
        yaw = psi + np.pi/2

        x += V_HORIZ * math.cos(psi) * dt
        y += V_HORIZ * math.sin(psi) * dt
        z = BASE_ALT + ALT_AMP * math.sin(ALT_W * t)

        # calculate acc
        ax = V_HORIZ * OMEGA * -math.sin(psi) + random.gauss(0, config.ACC_NOISE_STD)
        ay = V_HORIZ * OMEGA * math.cos(psi) + random.gauss(0, config.ACC_NOISE_STD)
        az = -ALT_AMP * ALT_W * ALT_W * math.sin(ALT_W * t) + random.gauss(0, config.ACC_NOISE_STD)

        ax, ay = rotate_vector(np.array([ax, ay]), -yaw)

        trans.setSFVec3f([x, y, z])
        rot.setSFRotation([0, 0, 1, psi])

        # --- synthesise the sensors ------------------------------------------
        gyro_z = OMEGA + config.GYRO_BIAS + random.gauss(0, config.GYRO_NOISE_STD)

        baro = None
        if since_baro >= config.BARO_PERIOD_S:
            since_baro = 0.0
            baro = z + random.gauss(0, config.BARO_NOISE_STD)

        imu_meas = np.array([ax, ay, az + 9.81, 0.0, 0.0, gyro_z])  # ax, ay, az, gx, gy, gz
        gps_xyz = None
        if since_gps >= config.GPS_PERIOD_S:
            since_gps = 0.0
            gps_xyz = (x + random.gauss(0, config.GPS_NOISE_XY),
                       y + random.gauss(0, config.GPS_NOISE_XY),
                       z + random.gauss(0, config.GPS_NOISE_Z))
            
        mag = None
        mag_yaw = None
        if since_mag >= config.MAG_PERIOD_S:
            since_mag = 0.0
            mag_yaw = yaw + random.gauss(0, config.MAG_NOISE_STD)
            mag = psi_to_mag(mag_yaw)

        camera = None
        if since_camera >= config.CAMERA_PERIOD_S:
            since_camera = 0.0
            # --- camera sees the world from the drone's perspective -----------
            # 2D transformation matrix in drone frame:
            scale = (last_vis_pz / z) + random.gauss(0, config.CAMERA_SCALE_STD)

            # world to drone frame rotation matrix
            R_world_to_drone = np.array([
                [math.cos(-yaw), -math.sin(-yaw)],
                [math.sin(-yaw),  math.cos(-yaw)]
            ])

            # compute the shift in world frame and then transform to drone frame
            world_shift = (np.array([x, y]) - np.array([last_vis_px, last_vis_py]))
            world_shift += np.random.normal(0, config.CAMERA_SHIFT_STD, size=2)

            drone_shift = R_world_to_drone @ world_shift
            drone_shift /= scale

            d_heading = (yaw - last_vis_heading) + random.gauss(0, config.CAMERA_ROT_STD)

            camera = np.zeros((2, 3))
            R = np.array([[math.cos(d_heading), -math.sin(d_heading)],
                            [math.sin(d_heading),  math.cos(d_heading)]])
            camera[:2, :2] = R / scale
            camera[:2, 2] = drone_shift

            last_vis_px += world_shift[0]
            last_vis_py += world_shift[1]
            last_vis_pz = z
            last_vis_heading = yaw

        sensors = {
            "t": t,
            "dt": dt,
            "imu": imu_meas,     # yaw rate (rad/s), biased
            "camera": camera,       # horizontal speed (m/s)
            "baro": baro,         # altitude (m): smooth + frequent, low noise
            "gps": gps_xyz,       # absolute (x, y, z) with noise, or None between updates
            "mag": mag  # heading (rad), smooth + frequent, low noise
        }

        est_x, est_y, est_z, est_yaw, health = estimator.estimate(sensors)

        # --- score against ground truth (3D) ---------------------------------
        writer.writerow([
            f"{t:.3f}", x, y, z, psi, est_x, est_y, est_z, est_yaw,
            "" if gps_xyz is None else gps_xyz[0],
            "" if gps_xyz is None else gps_xyz[1],
            "" if gps_xyz is None else gps_xyz[2],
            "" if camera is None else last_vis_px,
            "" if camera is None else last_vis_py,
            "" if camera is None else last_vis_pz,
            "" if camera is None else last_vis_heading,
            "" if baro is None else baro,
            "" if mag_yaw is None else mag_yaw,
            "" if health["gps"]["nis"] is None else health["gps"]["nis"],
            "" if health["baro"]["nis"] is None else health["baro"]["nis"],
            "" if health["mag"]["nis"] is None else health["mag"]["nis"],
            "" if health["camera"]["nis"] is None else health["camera"]["nis"],
            "" if health["gps"]["nis"] is None else health["gps"]["fault"],
            "" if health["baro"]["nis"] is None else health["baro"]["fault"],
            "" if health["mag"]["nis"] is None else health["mag"]["fault"],
            "" if health["camera"]["nis"] is None else health["camera"]["fault"]
        ])

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
