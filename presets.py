"""
Order-of-magnitude sensor parameters, so we stop guessing (README: "no idea about coherent values").

Datasheets give NOISE DENSITIES (per sqrt(Hz)); the simulator draws one sample every dt,
so the per-sample standard deviation is  sigma = density / sqrt(dt).
IMU2D draws its bias random walk as  bias += sigma_bias * sqrt(dt) * N(0,1),
so its sigma_bias_* are random-walk densities (unit per sqrt(s)).

Values are typical of each class, not of one product -- quote a real datasheet in the pitch
if you can. 1 ug = 9.81e-6 m/s^2 ; 1 deg/h = 4.85e-6 rad/s ; 1 deg/sqrt(h) = 2.91e-4 rad/sqrt(s).
"""
import numpy as np

UG = 9.81e-6
DEG_H = np.radians(1) / 3600
DEG_SQRTH = np.radians(1) / 60

IMU_GRADES = {
    #                accel noise      accel bias     accel bias RW     gyro ARW            gyro bias        gyro bias RW
    #                [m/s^2/sqrtHz]   [m/s^2]        [m/s^2/sqrt s]    [rad/s/sqrtHz]      [rad/s]          [rad/s/sqrt s]
    # ship / submarine INS (ring-laser or fibre-optic gyros); drifts ~1-2 km per hour alone
    "navigation": dict(acc_nd=10 * UG,  acc_bias=25 * UG,   acc_bias_rw=0.1 * UG,  gyro_nd=0.002 * DEG_SQRTH, gyro_bias=0.01 * DEG_H, gyro_bias_rw=1e-4 * DEG_H),
    # tactical grade (missiles, UAVs, small vessels); drifts tens of km per hour
    "tactical":   dict(acc_nd=50 * UG,  acc_bias=300 * UG,  acc_bias_rw=3 * UG,    gyro_nd=0.1 * DEG_SQRTH,   gyro_bias=1.0 * DEG_H,  gyro_bias_rw=0.01 * DEG_H),
    # consumer MEMS (phone, drone); useless alone after a few minutes
    "consumer":   dict(acc_nd=150 * UG, acc_bias=5000 * UG, acc_bias_rw=50 * UG,   gyro_nd=0.3 * DEG_SQRTH,   gyro_bias=50 * DEG_H,   gyro_bias_rw=1 * DEG_H),
}
# For reference, IMU2D's defaults (sigma_acc=0.03 at dt=0.1, bias 0.01 m/s^2, gyro bias 0.2 deg/s)
# are a low-cost MEMS -- worse than "consumer" for the gyro. Fine for tests, not for a ship.

GNSS = dict(sigma=3.0)          # m, civil single-frequency horizontal (1 sigma), open sky

# Atom-interferometer gravimeter, from the 2026 Coral Sea sea trial (Everitt et al., arXiv:2608.25563)
QUANTUM_GRAVIMETER = dict(noise_density_mgal=7.5,   # mGal/sqrt(Hz)
                          drift_mgal=0.7)           # mGal, slow drift (stability)


def imu_kwargs(grade: str, dt: float, seed: int = 42) -> dict:
    """Keyword arguments for IMU2D for a given grade and simulation step dt.
    Turn-on biases get a random sign/size so the filter cannot just 'know' them."""
    g = IMU_GRADES[grade]
    rng = np.random.default_rng(seed + 1000)
    return dict(
        sigma_acc=g["acc_nd"] / np.sqrt(dt),
        sigma_gyro=g["gyro_nd"] / np.sqrt(dt),
        acc_bias=tuple(rng.normal(0, g["acc_bias"], 2)),
        gyro_bias=float(rng.normal(0, g["gyro_bias"])),
        sigma_bias_accel=g["acc_bias_rw"],
        sigma_gyro_bias=g["gyro_bias_rw"],
        seed=seed,
    )
