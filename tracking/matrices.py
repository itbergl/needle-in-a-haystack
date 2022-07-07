import numpy as np
from ..globals import *

class Matrices():
    def __init__(self):
        """
            tau: time param
            sigmas: [sigma_p, sigma_v, sigma_a, sigma_d]
        """
        tau = g.hyperparams.tau
        sigmas = [g.hyperparams.sigmas[0], g.hyperparams.sigmas[1],
            g.hyperparams.sigmas[2], g.hyperparams.sigmas[3]]

        self.Fk = np.matrix([
            [1, 0, tau, 0, tau**2/2, 0],
            [0, 1, 0, tau, 0, tau**2/2],
            [0, 0, 1, 0, tau, 0],
            [0, 0, 0, 1, 0, tau],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.Hk = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # square each sigma and have two of each
        variances = [sigmas[int(i/2)]**2 for i in range(8)]

        # create covar matrices
        self.Qk = np.matrix(variances[:6] * np.identity(6))
        self.Rk = np.matrix(variances[6:] * np.identity(2))