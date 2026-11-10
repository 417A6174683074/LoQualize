from .movements.space import Movement2D#, Circular2d, Space2D
import numpy as np
from typing import Tuple
from .gravity import eotvos_mgal, eotvos_jacobian

class Measurement_device:
    def __init__(self) -> None:
        self.R = self.buildR()
        pass

    def observation(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def buildR(self):
        raise NotImplementedError

    



class IMU2D(Measurement_device):
    def __init__(self, movement: Movement2D,
                 sigma_acc:float =0.03,
                 sigma_gyro:float =np.deg2rad(0.1),
                 acc_bias:Tuple[float,float] =(0.01, -0.015),
                 gyro_bias:float = np.deg2rad(0.2),
                 sigma_gyro_bias: float= np.deg2rad(0.005),
                 sigma_bias_accel: float= 0.0002,
                 seed:int =42):
        self.rng = np.random.default_rng(seed)
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro
        self.sigma_bias_acc = sigma_bias_accel
        self.sigma_bias_gyro = sigma_gyro_bias
        self.movement: Movement2D = movement
        self.bias_acc = np.array(acc_bias, dtype=float)
        self.bias_gyro = gyro_bias
        self.R = self.buildR()

    def buildR(self)->np.ndarray:
        R = np.diag([self.sigma_acc**2]*2+[self.sigma_gyro**2])
        return R

    def observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_x,a_y,theta,omega,bx,by,bw=(state[i] for i in [4,5,6,7,8,9,10])
        c,s=np.cos(theta),np.sin(theta)
        h=np.array([c*a_x+s*a_y+bx,-s*a_x+c*a_y+by,omega+bw])
        H=np.zeros((3,len(state)))
        H[0,4]=c; H[0,5]=s; H[0,6]=-s*a_x+c*a_y; H[0,8]=1
        H[1,4]=-s; H[1,5]=c; H[1,6]=-c*a_x-s*a_y; H[1,9]=1
        H[2,7]=1; H[2,10]=1
        return h,H

    def angle(self)->float:
        vx,vy = self.movement.space.speed
        theta = np.arctan2(vy,vx)
        return theta

    def angle_speed(self)->float:
        vx, vy = self.movement.space.speed
        ax, ay = self.movement.space.acc
        nom = vx*ay - vy*ax
        denom = vx**2 + vy**2
        if denom < 1e-8:
            return 0.0
        return nom/denom


    def step(self):
        theta = self.angle()
        # print(theta)
        c, s = np.cos(theta), np.sin(theta)
        R_T = np.asarray([[c,s],[-s,c]])

        omega = self.angle_speed()
        accel_loc = R_T@self.movement.space.acc

        self.bias_acc += (
            self.sigma_bias_acc*np.sqrt(self.movement.dt)
            *
            self.rng.normal(size=2)
            )
        self.bias_gyro += (
                    self.sigma_bias_gyro*np.sqrt(self.movement.dt)
                    *
                    self.rng.normal()
                )

        noisy_acc = (accel_loc + self.bias_acc + self.rng.normal(0, self.sigma_acc, 2))
        noisy_gyro = (omega + self.bias_gyro + self.rng.normal(0, self.sigma_gyro))

        return noisy_acc, noisy_gyro    

class Satellite2D(Measurement_device):
    def __init__(self, movement: Movement2D, std_x: float, std_y: float, seed:int = 42):
        self.move = movement
        self.sigma_x = std_x
        self.sigma_y = std_y
        self.rng = np.random.default_rng(seed)
        self.R = self.buildR()

    def buildR(self):
        R = np.diag([self.sigma_x**2, self.sigma_y**2])
        return R

    def observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = state[:2]
        H = np.zeros((2,len(state)))
        H[0,0]=1
        H[1,1]=1
        return h, H

    def step(self):
        pos = self.move.space.position
        nois_pos = pos + self.rng.normal([0, 0], [self.sigma_x, self.sigma_y],2)
        return nois_pos


class Gravimeter2D(Measurement_device):
    """
    Quantum (atom-interferometer) gravimeter, as the navigation filter sees it.
    We do not simulate the atoms: navigation only sees "gravity + noise".

    step()            -> simulated reading: TRUE gravity at the TRUE position
                         + Eotvos effect of the TRUE velocity + slow drift + white noise,
                         averaged over `period` seconds (call it once every `period` s).
    observation(state)-> what the filter expects: MAP gravity at the ESTIMATED position
                         + Eotvos of the ESTIMATED velocity, and its Jacobian
                         H = [dg/dx, dg/dy, dE/dvx, dE/dvy, 0, ...].

    Defaults follow the 2026 Coral Sea trial (Everitt et al., arXiv:2608.25563).
    State layout expected: x[0:2] = position (east, north) [m], x[2:4] = velocity [m/s].
    """

    def __init__(self, movement: Movement2D, field,
                 noise_density_mgal: float = 7.5,   # mGal/sqrt(Hz)
                 drift_mgal: float = 0.7,           # mGal, slow drift (Gauss-Markov, std = drift/2)
                 drift_tau: float = 7200.0,         # s
                 period: float = 60.0,              # s, averaging window = one reading per period
                 lat_deg: float = -20.0,
                 corr_inflate: float = 3.0,         # map/unmapped/drift errors are correlated between
                                                    # consecutive readings -> inflate them in R
                 gate: float = 4.0,                 # innovation gate in sigmas (None = off)
                 seed: int = 42):
        self.movement = movement
        self.field = field
        self.nd = noise_density_mgal
        self.drift_mgal = drift_mgal
        self.drift_tau = drift_tau
        self.period = period
        self.lat = lat_deg
        self.corr_inflate = corr_inflate
        self.gate = gate
        self.rng = np.random.default_rng(seed)
        self.drift = 0.0
        self.every = max(1, int(round(period / movement.dt)))   # in simulation steps
        self.R = self.buildR()

    def buildR(self) -> np.ndarray:
        var = (self.nd**2 / self.period
               + self.corr_inflate * (self.field.map_err_mgal**2 + self.field.fine_rms_mgal**2
                                      + (self.drift_mgal / 2)**2))
        return np.array([[var]])

    def ready(self, k: int) -> bool:
        return k % self.every == 0

    def step(self) -> np.ndarray:
        phi = np.exp(-self.period / self.drift_tau)
        self.drift = phi * self.drift + (self.drift_mgal / 2) * np.sqrt(1 - phi**2) * self.rng.normal()
        x, y = self.movement.space.position
        vx, vy = self.movement.space.speed
        z = (self.field.true(x, y) + eotvos_mgal(vx, vy, self.lat) + self.drift
             + self.rng.normal(0, self.nd / np.sqrt(self.period)))
        return np.array([z])

    def observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        h = self.field.map(x, y) + eotvos_mgal(vx, vy, self.lat)
        gx, gy = self.field.map_gradient(x, y)
        ex, ey = eotvos_jacobian(vx, vy, self.lat)
        H = np.zeros((1, len(state)))
        H[0, 0], H[0, 1], H[0, 2], H[0, 3] = gx, gy, ex, ey
        return np.array([h]), H
