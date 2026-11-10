if __name__ == "__main__" and not __package__:
    # Allow running this file directly from an IDE.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
try:
    from .space import Space2D, Movement2D          # works whatever the repo folder is called
except ImportError:
    from my_world.movements.space import Space2D, Movement2D   # running this file directly (needs folder "my_world")
import matplotlib.pyplot as plt

class Sinus(Movement2D):
    def __init__(self, space: Space2D, amplitude: float, x_speed: float, pulsation: float=0) -> None:
        super().__init__(space)
        self.A = amplitude
        self.x_speed = x_speed
        self.omega = pulsation
        self.lambd = 2*np.pi*self.x_speed/self.omega

    def pos(self, t:float)->np.ndarray:
        return np.array([self.x_speed*t, self.A*np.sin(self.omega*t)])

    def speed(self, t:float)->np.ndarray:
        return np.array([self.x_speed, self.A*self.omega*np.cos(self.omega*t)])

    def acc(self, t: float)->np.ndarray:
        return np.array([0, -self.A*(self.omega**2)*np.sin(self.omega*t)])

    def orientation(self, speed: np.ndarray)->float:
        vx, vy = speed
        return np.arctan2(vy, vx)

    def ang_speed(self, speed: np.ndarray, acc: np.ndarray)->float:
        vx, vy = speed
        ax, ay = acc
        return (vx*ay-vy*ax)/(vx*vx+vy*vy)

    def step(self):
        self.space.position = self.pos(self.space.time)
        self.space.speed = self.speed(self.space.time)
        self.space.acc = self.acc(self.space.time)
        self.theta = self.orientation(self.space.speed)
        self.aspeed = self.ang_speed(self.space.speed, self.space.acc)
        self.space.time += self.space.ts



##-------------Test for the sinusoidal direction------------

def make_Q(G: np.ndarray, sigma_u, sigma_acc_bias_step, sigma_gyro_bias_step):
     sigma = np.diag(sigma_u**2)
     qb = np.diag([0,0,0,0,0,0,0,0,sigma_acc_bias_step**2,sigma_acc_bias_step**2,sigma_gyro_bias_step**2])
     return G@sigma@(G.T) + qb

if __name__ == "__main__":
    from my_world.sensors import IMU2D, Satellite2D
    from my_world.general_filt import Kalman

    rng = np.random.default_rng(42)
    space = Space2D()
    sinus = Sinus(space=space, amplitude=1000, x_speed=30, pulsation=np.pi/100)
    imu = IMU2D(movement=sinus)
    gps = Satellite2D(sinus, 3, 3)
    sinus.step()
    x0 = np.array([
            sinus.space.position[0],              # x
            sinus.space.position[1],                 # y
            sinus.space.speed[0],                 # vx
            sinus.space.speed[1],      # vy
            sinus.space.acc[0],  # ax
            sinus.space.acc[1],                 # ay
            np.pi / 2,           # theta
            sinus.aspeed,               # vitesse angulaire
            imu.bias_acc[0],             # biais accéléromètre x
            imu.bias_acc[1],             # biais accéléromètre y
            imu.bias_gyro            # biais gyroscope
        ])

    p0 = np.zeros((11,11)) # we suppose every initiale values to be perfectly known       
    #np.diag([0,0,0,0,0,0,0,0,imu.sigma_bias_acc,imu.sigma_bias_acc,imu.sigma_bias_gyro])

    sigma_u = np.asarray([0.01, 0.01, 0.001])

    F = np.eye(11)
    d=sinus.space.ts
    F[0,2]=d; F[1,3]=d
    F[0,4]=d*d/2; F[1,5]=d*d/2
    F[2,4]=d; F[3,5]=d
    F[6,7]=d

    G = np.zeros((11, 3))
    G[0,0] = G[1,1] = (d**3)/6
    G[2,0] = G[3,1] = G[6,2] = (d**2)/2
    G[4,0] = G[5,1] = G[7,2] = d

    Q = make_Q(G, sigma_u,imu.sigma_bias_acc,imu.sigma_bias_gyro)
    kalman_filt = Kalman(x0, p0, F=F, Q=Q, G=G, dt=d)

    positions = [sinus.space.position]
    filter_pos = [[kalman_filt.x[0], kalman_filt.x[1]]]
    error = [0]

    for i in range(6000):
            acc_before = np.array(sinus.space.acc, copy=True)
            omega_before = imu.angle_speed()
            sinus.step()
            acc_after = np.array(sinus.space.acc, copy=True)
            omega_after = imu.angle_speed()
            u_acc = (acc_after-acc_before)/d
            u_omega = (omega_after-omega_before)/d
            u = np.array([*u_acc,u_omega])
    
            u_noisy = u + np.random.normal([0,0,0], sigma_u, 3)
            # print(acc_before, acc_after)
            positions.append(sinus.space.position)
    
    
            device_acc, device_gyro = imu.step()
            measures = []
            devices = []
            z_acc = np.array([*np.asarray(device_acc), device_gyro])
            measures.append(z_acc)
            devices.append(imu)
            #to use GNNS positionning use:
            if i%1000 == 0 and i<=3000:
                z_gps = gps.step()
                measures.append(z_gps)
                devices.append(gps)
                
            kalman_filt.step(z=measures, devices=devices, u=u_noisy)
    
            filt_x, filt_y = kalman_filt.x[0], kalman_filt.x[1]
            filter_pos.append([filt_x, filt_y])
    
            err = np.sqrt(kalman_filt.P[0,0]+kalman_filt.P[1,1])
            error.append(err)
    
    
    
    
    positions = np.array(positions)
    filter_pos = np.array(filter_pos)
    # print(positions)
    figs, axs = plt.subplots(1)
    axs.plot(positions[:,0], positions[:,1], label="true pos")
    axs.plot(filter_pos[:,0], filter_pos[:,1], label="filter pos")
    

    # axs.plot(np.linspace(0,len(filter_pos),len(filter_pos)), np.sqrt((positions[:,0]-filter_pos[:,0])**2+(positions[:,1]-filter_pos[:,1])**2), label="position error")
    # axs.plot(np.linspace(0,len(filter_pos),len(filter_pos)), error, label="position sigma")
    plt.legend()
    plt.show()
