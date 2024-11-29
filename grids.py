import numpy as np

from particles import Particles


# 1 spatial index, 1 component
class Grid1D:
    def __init__(self, x_max, dx):
        self.x_max = x_max
        self.dx = dx
        self.x = np.arange(0, x_max, dx)
        self.n_cells = self.x.size
        # Cell averaged-quantities
        self.E_0 = np.zeros(self.n_cells)
        self.E = np.zeros(self.n_cells)
        self.n_e = np.empty(self.n_cells)
        self.n_i = np.empty(self.n_cells)
        self.rho = np.empty(self.n_cells)
        self.phi = np.empty(self.n_cells)

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        # NOTE TO SELF (Simon): I'm not fully convinced of how this functionality is split up from the CIC-interpolation, something to look further into...
        #   Maybe electrons.set_weights(); ions.set_weights(); grid.set_densities(); in main loop?
        dummy = electrons.x / self.dx
        electrons.idx = dummy.astype(int)
        electrons.cic_weights = dummy - electrons.idx
        self.n_e.fill(0)
        np.add.at(self.n_e, electrons.idx, 1 - electrons.cic_weights)
        # TODO: We're assuming periodic BC here, take into account params.bc!
        np.add.at(self.n_e, (electrons.idx + 1) % self.n_cells, electrons.cic_weights)

        dummy = ions.x / self.dx
        ions.idx = dummy.astype(int)
        ions.cic_weights = dummy - ions.idx
        self.n_i.fill(0)
        np.add.at(self.n_i, ions.idx, 1 - ions.cic_weights)
        np.add.at(self.n_i, (ions.idx + 1) % self.n_cells, ions.cic_weights)

        self.rho = electrons.q * self.n_e + ions.q * self.n_i
        # Remove mean charge, I'm not sure about the physical validity of doing this?
        self.rho -= np.mean(self.rho)


# 1 spatial index, 3 components, adding B and J
class Grid1D3V:
    def __init__(self, x_max, dx):
        self.x_max = x_max
        self.dx = dx
        self.x = np.arange(0, x_max, dx)
        self.n_cells = self.x.size
        # External fields:
        self.E_0 = np.zeros((self.n_cells, 3))
        self.B_0 = np.zeros((self.n_cells, 3))
        # Total fields
        self.E = np.zeros((self.n_cells, 3))
        self.E[:,1:] = self.E_0[:,1:] #the x component is always calculated directly using the poisson solver so it has to be added there each timestep
        self.J = np.zeros((self.n_cells, 3)) 
        self.B = np.zeros((self.n_cells, 3)) + self.B_0
        
        # Cell averaged-quantities
        self.n_e = np.empty(self.n_cells)
        self.n_i = np.empty(self.n_cells)
        self.rho = np.empty(self.n_cells)
        self.phi = np.empty(self.n_cells)

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        dummy = electrons.x / self.dx
        electrons.idx = dummy.astype(int)
        electrons.cic_weights = dummy - electrons.idx
        self.n_e.fill(0)
        np.add.at(self.n_e, electrons.idx, 1 - electrons.cic_weights)
        # TODO: We're assuming periodic BC here, take into account params.bc!
        np.add.at(self.n_e, (electrons.idx + 1) % self.n_cells, electrons.cic_weights)

        dummy = ions.x / self.dx
        ions.idx = dummy.astype(int)
        ions.cic_weights = dummy - ions.idx
        self.n_i.fill(0)
        np.add.at(self.n_i, ions.idx, 1 - ions.cic_weights)
        np.add.at(self.n_i, (ions.idx + 1) % self.n_cells, ions.cic_weights)

        self.rho = electrons.q * self.n_e + ions.q * self.n_i
        # Remove mean charge, I'm not sure about the physical validity of doing this?
        self.rho -= np.mean(self.rho)


# 2 spatial indices, 2/3 components
class Grid2D:
    def __init__(self):
        raise NotImplementedError
