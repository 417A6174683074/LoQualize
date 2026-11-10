"""Gravity world: the TRUE gravity anomaly field and the (imperfect) MAP the navigator carries.

Keep these two separate on purpose (same philosophy as the README: the filter must know less
than the simulator). The true field contains fine structure the map does not have, and the map
has its own error -- in practice map error, not gravimeter noise, is what limits accuracy.

Units: metres for positions, mGal for gravity (1 mGal = 1e-5 m/s^2).
Frame:  x = east, y = north (needed for the Eotvos effect).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

OMEGA_EARTH = 7.292115e-5   # rad/s
R_EARTH = 6.371e6           # m
MGAL = 1e-5                 # m/s^2


def eotvos_mgal(vx, vy, lat_deg):
    """Eotvos effect seen by a moving gravimeter (vx east, vy north, m/s) -> mGal."""
    return (2 * OMEGA_EARTH * np.cos(np.radians(lat_deg)) * vx + (vx**2 + vy**2) / R_EARTH) / MGAL


def eotvos_jacobian(vx, vy, lat_deg):
    """d(Eotvos)/d(vx), d(Eotvos)/d(vy) in mGal per m/s."""
    return ((2 * OMEGA_EARTH * np.cos(np.radians(lat_deg)) + 2 * vx / R_EARTH) / MGAL,
            (2 * vy / R_EARTH) / MGAL)


class GravityField:
    """
    terrain="rough": seamounts / trenches (informative for navigation)
    terrain="flat" : smooth, weak anomalies (map-matching should not help much)

    extent       : side of the square world [m], origin at (0, 0)
    dx_map       : map grid spacing, 1852 m = 1 arc-minute (Scripps / SWOT marine gravity grid)
    map_err_mgal : rms error of the map
    fine_rms_mgal: rms of real sub-km structure that is NOT in the map
    """

    def __init__(self, terrain="rough", extent=160e3, dx_true=250.0, dx_map=1852.0,
                 map_err_mgal=1.0, fine_rms_mgal=1.0, seed=0):
        rng = np.random.default_rng(seed)
        n = int(extent / dx_true) + 1
        xs = np.linspace(0, extent, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        rough = terrain == "rough"

        corr = 8e3 if rough else 30e3
        g = gaussian_filter(rng.standard_normal((n, n)), corr / dx_true, mode="wrap")
        g *= (15.0 if rough else 2.0) / g.std()
        if rough:
            for _ in range(22):
                cx, cy = rng.uniform(0, extent, 2)
                w = rng.uniform(3e3, 9e3)
                A = rng.uniform(25, 80) * rng.choice([1, 1, -0.6])
                g = g + A * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * w**2))
        fine = gaussian_filter(rng.standard_normal((n, n)), 2.0)
        fine *= fine_rms_mgal / fine.std()
        self._true = RectBivariateSpline(xs, xs, g + fine)

        # the map: low-passed to map resolution, sampled on the map grid, plus correlated error
        smooth = RectBivariateSpline(xs, xs, gaussian_filter(g, (dx_map / 2) / dx_true))
        nm = int(extent / dx_map) + 1
        self.xm = np.linspace(0, extent, nm)
        mp = smooth(self.xm, self.xm)
        err = gaussian_filter(rng.standard_normal(mp.shape), 0.7)
        mp += map_err_mgal * err / err.std()
        self.map_grid = mp
        self._map = RectBivariateSpline(self.xm, self.xm, mp)
        self.extent = extent
        self.map_err_mgal = map_err_mgal
        self.fine_rms_mgal = fine_rms_mgal

    # ground truth -- only the simulated SENSOR may call this
    def true(self, x, y):
        return self._true.ev(x, y)

    # what the navigator knows -- only the FILTER may call these
    def map(self, x, y):
        return self._map.ev(x, y)

    def map_gradient(self, x, y):
        """(dg/dx, dg/dy) of the map in mGal per metre."""
        return self._map.ev(x, y, dx=1), self._map.ev(x, y, dy=1)