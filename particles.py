import numpy as np
import numpy.typing as npt


class Particles:
    def __init__(self, num_particles: int, dimX: int, dimV: int, mass: float, charge: float):
        self.x = np.empty((num_particles, dimX))
        self.v = np.empty((num_particles, dimV))
        self.idx: npt.NDArray = None
        # TODO: add extra dimension to store weights for higher order b-splines?
        self.cic_weights: npt.NDArray = None
        self.m = mass
        self.q = charge
        self.qm = charge / mass
        self.N = num_particles
        self.dimX = dimX
        self.dimV = dimV

    def filter(self, mask: npt.NDArray):
        """
        Only keep the particles where mask[i] = True.
        mask should be a 1D boolean array of the same length as x and y
        """
        self.x = self.x[mask]
        self.v = self.v[mask]
        self.N = self.x.shape[0]

    def add_particles(self, x_new: npt.NDArray, v_new: npt.NDArray):
        """
        Add new particles with given speed and velocity
        """
        self.x = np.concatenate((self.x, x_new))
        self.v = np.concatenate((self.v, v_new))
        self.N = self.x.shape[0]

    def kinetic_energy(self):
        """
        Return the total kinetic energy of the particles
        KE = 1/2 * m * sum_{j=1}^{N} v_j**2
        """
        return 0.5 * self.m * np.sum(self.v**2)
