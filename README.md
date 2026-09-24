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


## Structure

```
project/
├── README.md            # this file
├── requirements.txt
├── main.py              # single entry point: load -> fuse -> evaluate -> plot
├── webots/drone/
│   ├── data_loader.py   # EuRoC parser + GNSS surrogate + interpolation
│   ├── ekf.py           # 15-state INS/GNSS EKF (numerical Jacobian)
│   └── visualization.py # labelled trajectory + error plots
├── data/README.md       # how to get EuRoC (data itself is gitignored)
└── results/             # generated figures + rmse.txt
```