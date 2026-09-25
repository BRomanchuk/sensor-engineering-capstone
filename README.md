# sensor-engineering-capstone
Sensor Engineering Capstone


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
```
Reading <path-to-trajectory.csv>
3D ATE (position RMSE): 0.090 m   (raw-GPS baseline ~ 1.415 m)
Terminal Error (3D distance to goal): 0.174 m
Total mission distance: 44.950 m
Saved <path-to-trajectory.png>
```

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