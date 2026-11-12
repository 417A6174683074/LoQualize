"""
Quantum-gravimeter-aided navigation, built on the repo's Kalman / IMU2D / Satellite2D.

Scenario: a ship sails for a few hours; GNSS works for the first 30 min, then is jammed.
Two filters see EXACTLY the same sensor readings:
    - "INS only"            : IMU (+ GNSS while available)
    - "INS + gravimeter"    : same, plus the quantum gravimeter matched against a gravity map

Run from the folder above the repo (relative imports, like main.py):
    python -m LoQualize.main_gravity                       # one run, seamounts, navigation-grade IMU
    python -m LoQualize.main_gravity --terrain flat
    python -m LoQualize.main_gravity --seeds 10 --no-show  # Monte Carlo, saves figures
    python -m LoQualize.main_gravity --grade tactical
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .movements import Space2D, ShipRoute
from .sensors import IMU2D, Satellite2D, Gravimeter2D
from .general_filt import Kalman
from .gravity import GravityField
from .models import make_F, make_Q_jerk, N_STATE
from .presets import IMU_GRADES, imu_kwargs, GNSS, QUANTUM_GRAVIMETER


def run(field, grade="navigation", hours=4.0, gnss_until=1800.0, gnss_every=10, dt=1.0,
        grav_period=60.0, seed=0, route_seed=1):
    space = Space2D(pos=np.zeros(2), speed=np.zeros(2), acc=np.zeros(2), time_step=dt)
    ship = ShipRoute(space, duration=hours * 3600, seed=route_seed)   # default: same route every run
    imu = IMU2D(movement=ship, **imu_kwargs(grade, dt, seed=seed))
    gps = Satellite2D(ship, GNSS["sigma"], GNSS["sigma"], seed=seed + 1)
    grav = Gravimeter2D(ship, field, noise_density_mgal=QUANTUM_GRAVIMETER["noise_density_mgal"],
                        drift_mgal=QUANTUM_GRAVIMETER["drift_mgal"], period=grav_period, seed=seed + 2)

    g = IMU_GRADES[grade]
    F = make_F(dt)
    Q = make_Q_jerk(dt, sigma_jerk=0.01, sigma_alpha=0.002,
                    acc_bias_rw=g["acc_bias_rw"], gyro_bias_rw=g["gyro_bias_rw"])
    G = np.zeros((N_STATE, 3))     # honest mode: no commanded jerk from the simulator
    u = np.zeros(3)

    rng = np.random.default_rng(seed + 3)
    x0 = np.r_[space.position + rng.normal(0, 10, 2), space.speed, space.acc,
               ship.theta, ship.aspeed, 0.0, 0.0, 0.0]          # biases unknown -> start at 0
    p0 = np.diag([10**2, 10**2, 0.1**2, 0.1**2, 0.01**2, 0.01**2, np.radians(1)**2, 1e-3**2,
                  g["acc_bias"]**2, g["acc_bias"]**2, g["gyro_bias"]**2])
    filters = {"ins": Kalman(x0.copy(), p0.copy(), F=F, Q=Q, G=G, dt=dt),
               "grav": Kalman(x0.copy(), p0.copy(), F=F, Q=Q, G=G, dt=dt)}

    n = ship.n_steps
    out = dict(t=np.zeros(n + 1), true=np.zeros((n + 1, 2)), gnss_until=gnss_until)
    for key in filters:
        out[key] = np.zeros((n + 1, 2))
        out[key + "_sig"] = np.zeros(n + 1)
    out["true"][0] = space.position
    for key, kf in filters.items():
        out[key][0] = kf.x[:2]

    for k in range(1, n + 1):
        ship.step()
        t = space.time
        acc, gyro = imu.step()
        meas, devs = [np.array([*acc, gyro])], [imu]
        if t <= gnss_until and k % gnss_every == 0:
            meas.append(gps.step()); devs.append(gps)
        meas_g, devs_g = list(meas), list(devs)
        if t > gnss_until and grav.ready(k):
            meas_g.append(grav.step()); devs_g.append(grav)

        filters["ins"].step(z=meas, devices=devs, u=u)
        filters["grav"].step(z=meas_g, devices=devs_g, u=u)

        out["t"][k] = t
        out["true"][k] = space.position
        for key, kf in filters.items():
            out[key][k] = kf.x[:2]
            out[key + "_sig"][k] = np.sqrt(kf.P[0, 0] + kf.P[1, 1])
    for key in filters:
        out[key + "_err"] = np.linalg.norm(out[key] - out["true"], axis=1)
    return out


LABEL = {"ins": "INS only", "grav": "INS + quantum gravimeter"}
COLOR = {"ins": "#2a78d6", "grav": "#eb6834"}


def plot(results, field, title, fname, show):
    t_h = results[0]["t"] / 3600
    k0 = np.searchsorted(results[0]["t"], results[0]["gnss_until"])
    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    r = results[0]
    xm = field.xm / 1e3
    im = ax_map.pcolormesh(xm, xm, field.map_grid.T, cmap="Greys", shading="auto")
    fig.colorbar(im, ax=ax_map, shrink=0.8).set_label("map gravity anomaly [mGal]")
    for other in results[1:]:      # other runs' routes (identical, hence invisible, unless --vary-route)
        ax_map.plot(*(other["true"].T / 1e3), color="#1f1f1d", lw=0.8, alpha=0.35)
    ax_map.plot(*(r["true"].T / 1e3), color="#1f1f1d", lw=2.5, label="true route (run 0)")
    for key in ("ins", "grav"):
        ax_map.plot(*(r[key][k0:].T / 1e3), color=COLOR[key], lw=2, label=LABEL[key])
    ax_map.plot(*(r["true"][k0] / 1e3), "o", color="#1f1f1d")
    ax_map.annotate("GNSS lost", r["true"][k0] / 1e3, xytext=(-62, -4), textcoords="offset points")
    pad = 12
    allp = np.vstack([q["true"] for q in results])
    ax_map.set_xlim(allp[:, 0].min() / 1e3 - pad, allp[:, 0].max() / 1e3 + pad)
    ax_map.set_ylim(allp[:, 1].min() / 1e3 - pad, allp[:, 1].max() / 1e3 + pad)
    ax_map.set_aspect("equal"); ax_map.set_xlabel("east [km]"); ax_map.set_ylabel("north [km]")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=2)

    for key in ("ins", "grav"):
        E = np.array([res[key + "_err"] for res in results]) / 1e3
        S = np.array([res[key + "_sig"] for res in results]) / 1e3
        med = np.median(E, 0)
        if len(results) > 1:
            lo, hi = np.percentile(E, [25, 75], 0)
            ax_err.fill_between(t_h, lo, hi, color=COLOR[key], alpha=0.15, lw=0)
        ax_err.plot(t_h, med, color=COLOR[key], lw=2, label=f"{LABEL[key]}: error")
        ax_err.plot(t_h, np.median(S, 0), color=COLOR[key], lw=1.2, ls="--",
                    label=f"{LABEL[key]}: filter's own 1-sigma")
        ax_err.annotate(f"{med[-1]:.1f} km", (t_h[-1], med[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9)
    ax_err.axvline(results[0]["gnss_until"] / 3600, color="#6b6a63", lw=1, ls=":")
    ax_err.set_xlabel("time [h]"); ax_err.set_ylabel("horizontal position error [km]")
    ax_err.set_title(f"median of {len(results)} run(s)" + (", band 25-75 %" if len(results) > 1 else ""),
                     loc="left", fontsize=10)
    ax_err.legend(loc="upper left", fontsize=8, frameon=False)
    for ax in (ax_err,):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e1e0d9", lw=0.6)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, format="png")
    print("saved", fname)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="rough", choices=["rough", "flat"])
    ap.add_argument("--grade", default="navigation", choices=list(IMU_GRADES))
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--vary-route", action="store_true", help="different ship route per seed (map panel: run 0 bold, others thin)")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    field = GravityField(a.terrain)
    results = []
    for s in range(a.seeds):
        r = run(field, grade=a.grade, hours=a.hours, seed=s, route_seed=1 + s * a.vary_route)
        print(f"seed {s}: final error  INS only {r['ins_err'][-1]:7.0f} m | "
              f"INS + gravimeter {r['grav_err'][-1]:7.0f} m")
        results.append(r)
    plot(results, field, f"GNSS lost after 30 min - {a.terrain} gravity, {a.grade}-grade IMU",
         f"gravity_{a.terrain}_{a.grade}.png", show=not a.no_show)