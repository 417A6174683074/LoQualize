import numpy as np
from .space import Space2D, Movement2D


class ShipRoute(Movement2D):
    """
    A ship-like route: long straight legs at constant speed joined by smooth turns
    (turn rate ramps up and down, so acceleration and jerk stay finite -- like an AIS track).

    Convention: x = east, y = north, heading measured like np.arctan2(vy, vx) (same as IMU2D.angle).
    The whole route is precomputed so position, speed and acceleration are mutually consistent.
    """

    def __init__(self, space: Space2D, duration: float = 4 * 3600, speed: float = 5.0,
                 start=(40e3, 40e3), heading_deg: float = 50.0, seed: int = 1) -> None:
        super().__init__(space)
        dt = self.dt
        rng = np.random.default_rng(seed)
        n = int(duration / dt) + 2

        # turn-rate profile: straight leg, then a turn with a 20 s ramp in and out
        r = np.zeros(n)
        k = 0
        while k < n:
            k += int(rng.uniform(15, 40) * 60 / dt)                  # straight leg
            dpsi = np.radians(rng.uniform(-70, 70))                  # heading change
            T = rng.uniform(60, 180)                                  # turn duration [s]
            ramp = min(20.0, T / 3)
            m = int(T / dt)
            t = np.arange(m) * dt
            shape = np.clip(np.minimum(t, T - t) / ramp, 0, 1)
            shape *= dpsi / (shape.sum() * dt)
            r[k:k + m] = shape[: max(0, min(m, n - k))]
            k += m

        psi = np.radians(heading_deg) + np.concatenate([[0], np.cumsum(0.5 * (r[1:] + r[:-1]) * dt)])
        v = speed * np.stack([np.cos(psi), np.sin(psi)], 1)
        a = speed * r[:, None] * np.stack([-np.sin(psi), np.cos(psi)], 1)
        p = np.empty_like(v)
        p[0] = start
        p[1:] = np.asarray(start) + np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)

        self._p, self._v, self._a, self._psi, self._r = p, v, a, psi, r
        self.k = 0
        self.n_steps = n - 1
        self._write()

    def _write(self):
        k = self.k
        self.space.position = self._p[k].copy()
        self.space.speed = self._v[k].copy()
        self.space.acc = self._a[k].copy()
        self.theta = self._psi[k]
        self.aspeed = self._r[k]

    def step(self):
        self.k += 1
        self._write()
        self.space.time += self.dt
