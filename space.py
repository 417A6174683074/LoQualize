from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Space2D:
    "Etat: p=[x, y], v=[vx, vy], a=[ax, ay]"

    def __init__(self, pos: np.ndarray=np.zeros(2), speed: np.ndarray=np.zeros(2),
                 acc: np.ndarray=np.zeros(2), g: float=-9.81, time_step: float=0.1) -> None:

        self.position: np.ndarray = pos
        self.speed: np.ndarray = speed
        self.acc: np.ndarray = acc
        self.time: float = 0
        self.g: float = g
        self.ts:float = time_step




class Movement2D(ABC):
    "indique le mouvement à effectuer"

    def __init__(self, space: Space2D) -> None:
        self.space = space
        self.dt = self.space.ts
    @abstractmethod
    def step(self):
        raise NotImplementedError


class Circular2d(Movement2D):
    "Ce mouvement consiste en un cercle autour de l'origine (0, 0)"
    def __init__(self, space: Space2D, r: float, ang_speed: float) -> None:
        super().__init__(space)

        self.rayon = r                  #m
        # self.ang_speed = ang_speed      #m/s
        self.space.position[0] = r
        self.space.position[1] = 0
        
        self.theta = ang_speed*self.dt
        self.aspeed = ang_speed
        self.c = np.cos(self.theta)
        self.s = np.sin(self.theta)

    def step(self):
        x, y = self.space.position
        # sx, sy = self.space.speed
        # ax, ay = self.space.acc


        nx = self.c*x-self.s*y
        ny = self.s*x+self.c*y

        nvx = -self.aspeed*ny
        nvy = self.aspeed*nx

        nax = -(self.aspeed**2)*nx
        nay = -(self.aspeed**2)*ny

        self.space.position = np.array((nx,ny))
        self.space.speed = np.array((nvx, nvy))
        self.space.acc = np.array((nax,nay))

        self.space.time += self.dt

if __name__ == "__main__":
    space = Space2D()
    circle = Circular2d(space=space, r=10, ang_speed=0.1)
    positions = [circle.space.position]
    for _ in range(627):
        circle.step()
        positions.append(circle.space.position)

    positions = np.array(positions)
    # print(positions)
    plt.plot(positions[:,0], positions[:,1])
    plt.show()