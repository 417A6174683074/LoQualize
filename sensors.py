from space import Movement2D#, Circular2d, Space2D
import numpy as np
from typing import Tuple

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