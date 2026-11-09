import numpy as np
from .sensors import Measurement_device

class Kalman:
    def __init__(self, x0: np.ndarray, p0: np.ndarray, F: np.ndarray, Q:np.ndarray, G:np.ndarray, dt: float=0.1):

        assert len(x0.shape)==1
        self.x = x0
        # assert len(p0.shape)==1
        self.P = p0
        assert len(F.shape)==2 and F.shape[0]==x0.shape[0] and F.shape[1]==x0.shape[0]
        self.F = F
        self.Q = Q
        self.G = G
        self.dt = dt


    def predict(self, u:np.ndarray):
        self.x = self.F@self.x + self.G@u
        self.P = self.F@self.P@self.F.T+self.Q
        self.P=(self.P+self.P.T)/2  # bcs it is suppose to be symmetric

    def update(self, z, h, H, R):
            y = z-h
            S = H@self.P@H.T+R
            K = np.linalg.solve(S, H@self.P).T
            self.x = self.x + K@y
            I = np.eye(11)
            A = I-K@H
            self.P=A@self.P@A.T+K@R@K.T  # Joseph
            self.P=(self.P+self.P.T)/2
            return y, K

    def step(self, z:list, devices: list[Measurement_device], u: np.ndarray):
         assert len(z)==len(devices)    #z contains the measures from all the devices
         self.predict(u)
         y, K = None, None
         hs = []
         Hs = []
         for measures, device in zip(z, devices):
            h, H=device.observation(state=self.x)
            R = device.R
            y, K = self.update(measures, h, H, R)
         return y, K




        
