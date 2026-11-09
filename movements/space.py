from abc import ABC, abstractmethod
import numpy as np
# import matplotlib.pyplot as plt


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

