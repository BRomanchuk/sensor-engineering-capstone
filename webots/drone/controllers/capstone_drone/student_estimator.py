"""THE ONLY FILE YOU EDIT for the drone world.

Contract:
    StudentEstimator.estimate(sensors) -> (x, y, z, yaw)

`sensors` is a dict, delivered every control tick:
    sensors["t"]      float  -- time since start (s)
    sensors["dt"]     float  -- time step (s)
    sensors["gyro_z"] float  -- yaw rate about z (rad/s); smooth but biased
    sensors["speed"]  float  -- HORIZONTAL speed (m/s)
    sensors["baro"]   float  -- altitude (m): smooth, frequent, low noise
    sensors["gps"]    (x, y, z) with noise, OR None between the ~1 Hz updates

Return your best estimate of the drone pose in the world frame:
    x, y, z  -- position (m)
    yaw      -- heading (rad), rotation about z

The third dimension is the new idea here. The GPS z is much noisier than its x/y
(as on real receivers), while the BAROMETER gives a smooth, frequent altitude.
Fuse baro (prediction/smoothing of z) with the occasional GPS z (absolute) and
you get an altitude far better than either alone -- that is where most of the ATE
gain is. For x/y, fuse gyro+speed dead-reckoning with the GPS as usual.

The DEFAULT below just echoes the last GPS fix (a baseline). Replace it with a
filter (KF/EKF/UKF) that beats it.
"""

import math
import numpy as np

from ekf import MultiRateEKF

from scenarios import USE_GPS, USE_BARO, USE_MAG, USE_CAMERA
import sensors_config

class StudentEstimator:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        acc_noise = sensors_config.ACC_NOISE_STD
        gyro_noise = sensors_config.GYRO_NOISE_STD
        dt = 0.016

        # initial state covariance and process noise covariance matrices
        P0 = np.diag([1e-5, 1e-5, 1e-5, 1e-1, 1e-1, 1e-1, 1e-5])
        Q = np.diag([
            (0.5 * acc_noise * dt**2)**2, 
            (0.5 * acc_noise * dt**2)**2, 
            (0.5 * acc_noise * dt**2)**2,
            (acc_noise * dt)**2,
            (acc_noise * dt)**2,
            (acc_noise * dt)**2,
            (gyro_noise**2 * dt)
        ]) * 1.1

        # initialize the multi-rate EKF with the initial state, covariance, and process noise
        self.ekf = MultiRateEKF(
            x0=[0.0, -6.0, 4, 0.0, 0.0, 0.0, +np.pi/2],
            P0=P0,
            Q=Q
        )

    def estimate(self, sensors):
        # predict state
        self.ekf.predict(sensors["imu"], sensors["dt"])

        # get sensor readings
        gps = sensors["gps"]
        baro = sensors["baro"]
        mag = sensors["mag"]
        camera = sensors["camera"]

        # update state with available sensor readings
        health = {
            'gps': {"nis": None, "fault": False},
            'baro': {"nis": None, "fault": False},
            'mag': {"nis": None, "fault": False},
            'camera': {"nis": None, "fault": False},
        }
        if gps is not None and USE_GPS:
            health['gps']['nis'], health['gps']['fault'] = self.ekf.update('gps', np.array(gps))
        if baro is not None and USE_BARO:
            health['baro']['nis'], health['baro']['fault']  = self.ekf.update('baro', np.array([baro]))
        if mag is not None and USE_MAG:
            health['mag']['nis'], health['mag']['fault']  = self.ekf.update('mag', np.array(mag))
        if camera is not None and USE_CAMERA:
            health['camera']['nis'], health['camera']['fault']  = self.ekf.update('camera', camera)

        
        
        # extract position and heading from the EKF state
        self.x, self.y, self.z = self.ekf.position
        self.yaw = self.ekf.heading
        return self.x, self.y, self.z, self.yaw, health




# ---------------------------------------------------------------------------
# Reference sketch (split the problem):
#
#   Horizontal (x, y, yaw) -- same EKF as the ground robot:
#       predict: yaw += gyro_z*dt; x += speed*cos(yaw)*dt; y += speed*sin(yaw)*dt
#       update with GPS (x, y) when available.
#
#   Vertical (z, optionally vz) -- a tiny 1-D filter:
#       predict z from the previous z (or a vz state); it barely changes per tick.
#       correct z with the BARO every tick (low noise) and with the GPS z when
#       available (high noise -> small gain). Result: smooth, accurate altitude.
#
# Fusing baro + gps-z is where this world is won.
# ---------------------------------------------------------------------------
