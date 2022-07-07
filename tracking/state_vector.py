import numpy as np
from ..globals import *

default_acceleration = (0, 0)
default_velocity = (0, 0)

class StateVector():
    def __init__(self, centroids, width, height):
        # state vector
        self.xk = np.matrix([*centroids,  *default_velocity, *default_acceleration]).getT()

        # save dimensions of object based on initial hypothesis
        self.width = width
        self.height = height

        self.Pk = g.m.Qk.copy()

        self.prev_hypo_centroid = centroids

    def update_priori(self):
        """
            Updates this state vector and covariance based on its velocity and acceleration
        """
        self.xk = g.m.Fk * self.xk
        self.Pk = g.m.Fk * self.Pk * (g.m.Fk.getT()) + g.m.Qk

    def update_posteriori(self, kalman_gain_k, innovation_k):
        """
            updates the positions based on hypotheses centroids
        """
        self.xk = self.xk + kalman_gain_k * innovation_k
        self.Pk = (np.matrix(np.identity(6)) - kalman_gain_k * g.m.Hk) * self.Pk

