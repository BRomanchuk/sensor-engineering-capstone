import numpy as np
import config

# Sensor noise std devs
NOISE = {
    'acc':      config.ACC_NOISE_STD,    # m/s²
    'gyro':     config.GYRO_NOISE_STD,  # rad/s
    'gps_xy':   config.GPS_NOISE_XY,    # m
    'gps_z':    config.GPS_NOISE_Z,    # m
    'baro':     config.BARO_NOISE_STD,    # m
    'mag':      config.MAG_NOISE_STD,    # rad
    'camera_heading':   config.CAMERA_ROT_STD,  # rad
    'camera_shift':     config.CAMERA_SHIFT_STD,  # m
    'camera_scale':     config.CAMERA_SCALE_STD,  # m
}

def build_sensor_matrices(noise: dict = NOISE) -> dict:
    """
    Build H and R matrices for each sensor.

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
    """Extended Kalman Filter with asynchronous multi-sensor updates."""

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

    def initialize_from_gps(self, gps: np.ndarray) -> None:
        """Initialize the filter state with the first GPS and barometer readings."""
        self.x[:3] = gps
        self.last_vis_px = self.x[0]
        self.last_vis_py = self.x[1]
        self.last_vis_pz = self.x[2]
        self.initialized = True

    def predict(self, imu_meas: np.ndarray, dt: float) -> None:
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

        output: x_vis, y_vis, z_vis, heading
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
        return np.array([heading])  # return as array for consistency with z
    
    def _R_camera(self, R_nominal, prev_visual_z, scale_noise):
        abs_z_noise = np.abs(prev_visual_z) * scale_noise
        R_nominal[2, 2] = abs_z_noise ** 2
        return R_nominal

    def update(self, sensor_name: str, z: np.ndarray) -> float:
        """
        Update step for a given sensor.

        Args:
            sensor_name: 'gps', 'baro', or 'mag'
            z:           Measurement vector for the sensor

        Returns:
            NIS (Normalized Innovation Squared)
        """
        if sensor_name == 'mag':
            z = self._mag_to_heading(z)
        if sensor_name == 'camera':
            z = self._parse_transform_matrix(z)
        H = self.sensor_matrices[sensor_name]['H']
        R = self.sensor_matrices[sensor_name]['R']
        # if sensor_name == 'camera':
        #     # transform scale noise into absolute z noise based on last visual z
        #     R = self._R_camera(R, self.last_vis_pz, NOISE['camera_scale'])

        y = z - H @ self.x
        if sensor_name == 'mag':
            y = (y + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        NIS = y.T @ np.linalg.inv(S) @ y

        if sensor_name == 'camera':
            self.last_vis_px = z[0]
            self.last_vis_py = z[1]
            # self.last_vis_pz = z[2]
            self.last_vis_heading = z[2]

        return NIS

    @property
    def position(self) -> np.ndarray:
        """Return current position estimate [px, py, pz]."""
        return self.x[:3]

    @property
    def heading(self) -> float:
        """Return current heading (yaw) estimate in radians -- state x[6]."""
        return self.x[6]