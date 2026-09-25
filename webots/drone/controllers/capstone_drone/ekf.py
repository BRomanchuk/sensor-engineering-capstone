import numpy as np
import sensors_config as sensors_config

from scipy.stats import chi2


# NIS thresholds for fault detection
NIS_THRESHOLDS = {
   "gps": chi2.ppf(0.99, df=3),
    "baro": chi2.ppf(0.99, df=1),
    "camera": chi2.ppf(0.99, df=3),
    "mag": chi2.ppf(0.99, df=1),
}


# Sensor noise std devs
NOISE = {
    'acc':      sensors_config.ACC_NOISE_STD,    # m/s²
    'gyro':     sensors_config.GYRO_NOISE_STD,  # rad/s
    'gps_xy':   sensors_config.GPS_NOISE_XY,    # m
    'gps_z':    sensors_config.GPS_NOISE_Z,    # m
    'baro':     sensors_config.BARO_NOISE_STD,    # m
    'mag':      sensors_config.MAG_NOISE_STD,    # rad
    'camera_heading':   sensors_config.CAMERA_ROT_STD,  # rad
    'camera_shift':     sensors_config.CAMERA_SHIFT_STD,  # m
    'camera_scale':     sensors_config.CAMERA_SCALE_STD,  # m
}


def build_sensor_matrices(noise: dict = NOISE) -> dict:
    """
    Build H and R matrices for each sensor.
    
    Args:
        noise: Dictionary of sensor noise std devs.
    
    Returns:
        Dictionary with keys 'gps', 'baro', 'mag', 'camera', each containing
        a dict with keys 'H' and 'R'.
    """
    H_gps = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ])
    R_gps = np.diag([noise['gps_xy']**2, noise['gps_xy']**2, noise['gps_z']**2])

    H_baro = np.array([[0, 0, 1, 0, 0, 0, 0]])
    R_baro = np.array([[noise['baro']**2]])

    H_mag = np.array([[0, 0, 0, 0, 0, 0, 1]])
    R_mag = np.array([[noise['mag']**2]])

    H_camera = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        # [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    position_noise = noise['camera_shift'] * (1 + noise['camera_scale'])
    R_camera = np.diag([
        position_noise**2,
        position_noise**2,
        # noise['camera_shift']**2,
        noise['camera_heading']**2
    ])

    return {
        'gps': {'H': H_gps, 'R': R_gps},
        'baro': {'H': H_baro, 'R': R_baro},
        'mag': {'H': H_mag, 'R': R_mag},
        'camera': {'H': H_camera, 'R': R_camera}
    }


class MultiRateEKF:
    """Extended Kalman Filter with asynchronous multi-sensor updates and fault detection"""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray):
        """
        Initialize EKF.

        Args:
            x0:              Initial state, shape (7,)
            P0:              Initial covariance, shape (7, 7)
            Q:               Process noise covariance, shape (7, 7)
        """
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.sensor_matrices = build_sensor_matrices()
        self.initialized = False
        self.last_vis_px = x0[0]
        self.last_vis_py = x0[1]
        self.last_vis_pz = x0[2]
        self.last_vis_heading = x0[6]

    def predict(self, imu_meas: np.ndarray, dt: float) -> None:
        """
        Predict the next state and covariance based on IMU measurements.

        Args:
            imu_meas: IMU measurements [ax, ay, az, gx, gy, gz], shape (6,)
            dt:       Time step in seconds
        """
        x_pred = self._predict_state(imu_meas, dt)
        F = self._jacobian_F(imu_meas, dt)
        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q

    def _predict_state(self, imu_meas: np.ndarray, dt: float) -> np.ndarray:
        px, py, pz, vx, vy, vz, heading = self.x
        ax, ay, az, gx, gy, gz = imu_meas
        ax_w = ax * np.cos(heading) - ay * np.sin(heading)
        ay_w = ax * np.sin(heading) + ay * np.cos(heading)
        x_pred = np.array([
            px + vx * dt,
            py + vy * dt,
            pz + vz * dt,
            vx + ax_w * dt,
            vy + ay_w * dt,
            vz + (az - 9.81) * dt,
            heading + gz * dt
        ])
        return x_pred
    
    def _jacobian_F(self, imu_meas: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute the Jacobian of the state transition function with respect to the state.

        Args:
            imu_meas: IMU measurements [ax, ay, az, gx, gy, gz], shape (6,)
            dt:       Time step in seconds
        """
        px, py, pz, vx, vy, vz, heading = self.x
        ax, ay, az, gx, gy, gz = imu_meas
        dvx_dheading = (-ax * np.sin(heading) - ay * np.cos(heading)) * dt
        dvy_dheading = (ax * np.cos(heading) - ay * np.sin(heading)) * dt
        F = np.array([
            [1, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, dvx_dheading],
            [0, 0, 0, 0, 1, 0, dvy_dheading],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        return F
    
    def _parse_transform_matrix(self, T: np.ndarray) -> tuple:
        """
        Parse a 2x3 transformation matrix from camera to world frame.

        Args:
            T: 2x3 transformation matrix
        Returns:
            (x_vis, y_vis, heading): Estimated position and heading in world frame
        """
        scale = np.linalg.det(T[:2, :2]) ** 0.5
        shift_body = T[:2, 2]
        d_heading = np.arctan2(T[1, 0], T[0, 0])

        heading = self.last_vis_heading + d_heading

        R_body_to_world = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)]
        ])
        shift_world = R_body_to_world @ shift_body * scale

        x_vis = self.last_vis_px + shift_world[0]
        y_vis = self.last_vis_py + shift_world[1]
        # z_vis = self.last_vis_pz / scale

        return np.array([x_vis, y_vis, heading])

    def _mag_to_heading(self, mag_meas: np.ndarray) -> float:
        """
        Convert magnetometer measurement to heading (yaw) in radians.

        Args:
            mag_meas: [mx, my, mz], shape (3,)

        Returns:
            heading in radians
        """
        mx, my, mz = mag_meas
        heading = np.arctan2(my, mx)
        print("Magnetometer Heading (rad):", heading)
        return np.array([heading])  # return as array for consistency with z
    
    def _R_camera(self, R_nominal, prev_visual_z, scale_noise):
        """
        Adjust the camera measurement noise covariance based on the previous visual z and scale noise.

        Args:
            R_nominal: Nominal camera measurement noise covariance matrix
            prev_visual_z: Previous visual z measurement
            scale_noise: Scale noise standard deviation

        Returns:
            Adjusted camera measurement noise covariance matrix
        """
        abs_z_noise = np.abs(prev_visual_z) * scale_noise
        R_nominal[2, 2] = abs_z_noise ** 2
        return R_nominal

    def update(self, sensor_name: str, z: np.ndarray) -> float:
        """
        Update step for a given sensor.

        Args:
            sensor_name: Name of the sensor ('gps', 'baro', 'mag', 'camera')
            z:           Measurement vector for the sensor

        Returns:
            nis (float): Normalized Innovation Squared
            fault_detected (bool): True if fault detected, False otherwise
        """
        # preprocess magnetometer and camera measurements
        if sensor_name == 'mag':
            z = self._mag_to_heading(z)
        if sensor_name == 'camera':
            z = self._parse_transform_matrix(z)
        
        # get sensor matrices
        H = self.sensor_matrices[sensor_name]['H']
        R = self.sensor_matrices[sensor_name]['R']

        # adjust camera measurement noise based on last visual z and scale noise
        # if sensor_name == 'camera':
        #     # transform scale noise into absolute z noise based on last visual z
        #     R = self._R_camera(R, self.last_vis_pz, NOISE['camera_scale'])

        # compute innovation and NIS, detect faults
        y = z - H @ self.x
        if sensor_name == 'mag':
            y = (y + np.pi) % (2 * np.pi) - np.pi
            print("Magnetometer Innovation (rad):", y)

        S = H @ self.P @ H.T + R

        nis = y.T @ np.linalg.inv(S) @ y
        fault_detected = nis > NIS_THRESHOLDS[sensor_name]

        if fault_detected:
            # apply fallback: increase R for this update
            R_degraded = R * 10.0
            S = H @ self.P @ H.T + R_degraded

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        
        # update last visual position and heading
        if sensor_name == 'camera':
            self.last_vis_px = z[0]
            self.last_vis_py = z[1]
            # self.last_vis_pz = z[2]
            self.last_vis_heading = z[2]

        return nis, fault_detected

    @property
    def position(self) -> np.ndarray:
        """Return current position estimate [px, py, pz]."""
        return self.x[:3]

    @property
    def heading(self) -> float:
        """Return current heading (yaw) estimate in radians -- state x[6]."""
        return self.x[6]