"""Process model shared by the scenarios (same 11-state layout as main.py / circle.py):

    x = [x, y, vx, vy, ax, ay, theta, omega, b_ax, b_ay, b_gyro]

"Honest" mode: the filter does NOT get the commanded jerk u from the simulator (that would leak
ground truth). Acceleration/turn rate evolve as random walks (white jerk / angular acceleration)
and are corrected by the IMU measurement update. This is the _make_Q idea already in circle.py.
"""
import numpy as np

N_STATE = 11


def make_F(dt: float) -> np.ndarray:
    F = np.eye(N_STATE)
    F[0, 2] = F[1, 3] = dt
    F[0, 4] = F[1, 5] = dt * dt / 2
    F[2, 4] = F[3, 5] = dt
    F[6, 7] = dt
    return F


def make_Q_jerk(dt: float, sigma_jerk: float, sigma_alpha: float,
                acc_bias_rw: float, gyro_bias_rw: float) -> np.ndarray:
    """sigma_jerk [m/s^3], sigma_alpha [rad/s^2]: per-step white jerk / angular acceleration.
    acc_bias_rw, gyro_bias_rw: random-walk DENSITIES (per sqrt(s)), so variance per step = rw^2 * dt
    (main.py's make_Q uses rw^2 without dt, which is only right for dt = 1 s)."""
    L = np.zeros((N_STATE, 6))
    for p, v, a, j in [(0, 2, 4, 0), (1, 3, 5, 1)]:
        L[p, j] = dt**3 / 6
        L[v, j] = dt**2 / 2
        L[a, j] = dt
    L[6, 2] = dt * dt / 2
    L[7, 2] = dt
    L[8, 3] = L[9, 4] = L[10, 5] = 1
    D = np.diag([sigma_jerk**2, sigma_jerk**2, sigma_alpha**2,
                 acc_bias_rw**2 * dt, acc_bias_rw**2 * dt, gyro_bias_rw**2 * dt])
    return L @ D @ L.T
