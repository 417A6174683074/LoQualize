import matplotlib.pyplot as plt
import numpy as np
from space import Space2D, Circular2d
from sensors import IMU2D, Satellite2D
from general_filt import Kalman

"""for detailed explanation on kalman filters check
Fundamentals of Kalman Filtering: A Practical Approach
Third Edition
"""

def _make_Q(d, sigma_jerk, sigma_alpha, sigma_acc_bias_step, sigma_gyro_bias_step):
   
        L=np.zeros((11,6))
        for p,v,a,j in [(0,2,4,0),(1,3,5,1)]:
            L[p,j]=(d**3)/6
            L[v,j]=(d**2)/2
            L[a,j]=d
        L[6,2]=d*d/2
        L[7,2]=d
        L[8,3]=L[9,4]=L[10,5]=1
  
        D=np.diag([sigma_jerk**2,sigma_jerk**2,sigma_alpha**2,
                   sigma_acc_bias_step**2,sigma_acc_bias_step**2,sigma_gyro_bias_step**2])
        return L@D@L.T

#We have two different make_Q functions for tests with and without knowledge of the commands
def make_Q(G: np.ndarray, sigma_u, sigma_acc_bias_step, sigma_gyro_bias_step):
     sigma = np.diag(sigma_u**2)
     qb = np.diag([0,0,0,0,0,0,0,0,sigma_acc_bias_step**2,sigma_acc_bias_step**2,sigma_gyro_bias_step**2])
     return G@sigma@(G.T) + qb


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    space = Space2D()
    circle = Circular2d(space=space, r=1000, ang_speed=0.01)  #creation of the movement
    #initialisation of the space caracteristic (could be a function of circle)
    circle.space.speed[1] = circle.rayon * circle.aspeed
    circle.space.acc[0] = -circle.rayon * circle.aspeed**2

    #creation of the different measurement device
    device = IMU2D(movement=circle)#,sigma_bias_accel=0,acc_bias=(0,0))#,sigma_gyro_bias=0,gyro_bias=0)#
    gps = Satellite2D(circle, 3, 3)

    #Initial state
    x0 = np.array([
        circle.rayon,              # x
        0.0,                 # y
        0.0,                 # vx
        circle.rayon * circle.aspeed,      # vy
        -circle.rayon * circle.aspeed**2,  # ax
        0.0,                 # ay
        np.pi / 2,           # theta
        circle.aspeed,               # vitesse angulaire
        device.bias_acc[0],             # biais accéléromètre x
        device.bias_acc[1],             # biais accéléromètre y
        device.bias_gyro            # biais gyroscope
    ])

    #Initial covariance matrix
    p0 = np.diag([0,0,0,0,0,0,0,0,device.sigma_bias_acc,device.sigma_bias_acc,device.sigma_bias_gyro])
    # Q = _make_Q(d = circle.dt, sigma_jerk=2, sigma_alpha=2, 
    #             sigma_acc_bias_step=device.sigma_bias_acc, sigma_gyro_bias_step=device.sigma_bias_gyro)

    #Errors of commands (could change during the movement)
    sigma_u = np.asarray([0.01, 0.01, 0.001])#(0.01, 0.01, 0.001)

    #next-step matrix
    F = np.eye(11)
    d=circle.dt
    F[0,2]=d; F[1,3]=d
    F[0,4]=d*d/2; F[1,5]=d*d/2
    F[2,4]=d; F[3,5]=d
    F[6,7]=d
    
    #command integration
    G = np.zeros((11, 3))
    G[0,0] = G[1,1] = (d**3)/6
    G[2,0] = G[3,1] = G[6,2] = (d**2)/2
    G[4,0] = G[5,1] = G[7,2] = d
    # print(G)

    #covariance of the process errors
    Q = make_Q(G, sigma_u,device.sigma_bias_acc,device.sigma_bias_gyro)

    #creation of the filter
    kalman_filt = Kalman(x0, p0, F=F, Q=Q, G=G, dt=d)

    
    
    positions = [circle.space.position]
    filter_pos = [[kalman_filt.x[0], kalman_filt.x[1]]]
    error = [0]
    for i in range(6000):
        acc_before = np.array(circle.space.acc, copy=True)
        omega_before = device.angle_speed()
        circle.step()
        acc_after = np.array(circle.space.acc, copy=True)
        omega_after = device.angle_speed()
        u_acc = (acc_after-acc_before)/d
        u_omega = (omega_after-omega_before)/d
        u = np.array([*u_acc,u_omega])

        u_noisy = u + np.random.normal([0,0,0], sigma_u, 3)
        # print(acc_before, acc_after)
        positions.append(circle.space.position)


        device_acc, device_gyro = device.step()
        measures = []
        devices = []
        z_acc = np.array([*np.asarray(device_acc), device_gyro])
        measures.append(z_acc)
        devices.append(device)
        #to use GNNS positionning use:
        # if i%100 == 0:
        #     z_gps = gps.step()
        #     measures.append(z_gps)
        #     devices.append(gps)
            
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
