GPS_PERIOD_S = 1.0
GPS_NOISE_XY = 0.40      # metres, horizontal GPS noise
GPS_NOISE_Z = 1.20      # metres, VERTICAL GPS noise (much worse -- realistic)

BARO_NOISE_STD = 0.15   # metres, barometer/altimeter noise (smooth, frequent)
BARO_PERIOD_S = 0.1           # seconds, barometer update period (smooth, frequent)

GYRO_BIAS = 0.006
GYRO_NOISE_STD = 0.01
ACC_NOISE_STD = 0.1

CAMERA_SHIFT_STD = 0.1  # metres, camera pixel shift noise (smooth, frequent)
CAMERA_ROT_STD = 0.0005      # radians, camera rotation noise (smooth, frequent)
CAMERA_SCALE_STD = 0.001
CAMERA_PERIOD_S = 0.03           # seconds, camera update period (smooth, frequent)

MAG_NOISE_STD = 0.001      # radians, magnetometer noise (smooth, frequent)
MAG_PERIOD_S = 0.5             # seconds, magnetometer update period (smooth, frequent)