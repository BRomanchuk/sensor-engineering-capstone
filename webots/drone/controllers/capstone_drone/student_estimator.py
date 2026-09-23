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


class StudentEstimator:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        P0 = 1e-3 * np.eye(7)
        Q = 1e-4 * np.eye(7)
        self.ekf = MultiRateEKF(
            x0=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Initial state: [x, y, z, vx, vy, vz, yaw]
            P0=P0,  # Initial covariance
            Q=Q  # Process noise covariance
        )

    def estimate(self, sensors):
        # ---- DEFAULT baseline: hold the last GPS fix (REPLACE THIS) ----------

        if not self.ekf.initialized:
            gps = sensors["gps"]
            if gps is not None:
                self.ekf.initialize_from_gps(np.array(gps))
                self.x, self.y, self.z = self.ekf.position
                self.yaw = self.ekf.heading
                return self.x, self.y, self.z, self.yaw
            

        self.ekf.predict(sensors["imu"], sensors["dt"])

        gps = sensors["gps"]
        baro = sensors["baro"]
        mag = sensors["mag"]
        camera = sensors["camera"]
        if gps is not None:
            self.ekf.update('gps', np.array(gps))
        if baro is not None:
            self.ekf.update('baro', np.array([baro]))
        if mag is not None:
            self.ekf.update('mag', np.array(mag))
        if camera is not None:
            self.ekf.update('camera', camera)
        
        self.x, self.y, self.z = self.ekf.position
        self.yaw = self.ekf.heading
        return self.x, self.y, self.z, self.yaw




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
