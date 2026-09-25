# GNSS + Visual-Inertial Navigation for UAV 
A complete, runnable realization of a course capstone: an end-to-end sensor-fusion pipeline of GNSS + Visual-Inertial Navigation for UAV from raw simulated data to an evaluated state estimate, entirely on a Webots simulator, no hardware. 

**What it does:** fuses the **simulated Webots IMU readings** (accelerometer + gyroscope, 62.5 Hz)
with a **GNSS** receiver (position, 1 Hz), **Magnetometer** (heading, 2 Hz), **Barometer** (altitude, 10 Hz), **Camera** (heading and position change, 33 Hz) using a **7-state
VINS/GNSS Extended Kalman Filter** (position, velocity, heading). It evaluates the estimate against ground truth position (ATE / RMSE over the whole trajectory & terminal error) and plots it. You can enable/disable any sensor in config file `webots/drone/controllers/capstone_drone/scenarios.py` and evaluate resulting setup in simulator.

## Run

### 0. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # install required packages
```

### 1. Simulation
1. Start Webots simulator.
2. Open drone environment (`webots/drone/worlds/capstone_drone.wbt`)
3. Click Run

### 2. Evaluation
```bash
cd webots/drone
python evaluate.py
```

## Expected output

Console prints a summary and `webots/drone/controllers/capstone_drone/results/` gets a figure and csv table
```
Reading <path-to-trajectory.csv>
3D ATE (position RMSE): 0.090 m   (raw-GPS baseline ~ 1.415 m)
Terminal Error (3D distance to goal): 0.174 m
Total mission distance: 44.950 m
Saved <path-to-trajectory.png>
```

- `.../results/trajectory-<sensor1>-...-<sensorN>.png` -- horizontal and vertical position, heading and NIS graphs for a given set of sensors

## Structure

```
project/
├── README.md            # this file
├── requirements.txt
├── main.py              # single entry point: load -> fuse -> evaluate -> plot
└── webots/drone/
    ├── controllers/capstone_drone/   # EuRoC parser + GNSS surrogate + interpolation
    │   ├── sensors_config.py   # sensors noise and frequency config
    │   ├── scenarios.py        # config file with scenarios (enable/disable certain sensors)
    │   ├── capstone_drone.py   
    │   ├── ekf.py              # 7-state VINS/GNSS EKF
    │   ├── student_estimator.py   
    │   └── results/    # generated figure and .csv
    ├── ekf.py          # 15-state INS/GNSS EKF (numerical Jacobian)
    └── worlds/         # Webots worlds

```