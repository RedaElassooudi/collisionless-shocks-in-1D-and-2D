import numpy as np

from particles import Particles


# 1 spatial index, 1 component
class Grid1D:
    def __init__(self, x_max, dx):
        self.x_max = x_max
        self.dx = dx
        self.x = np.arange(0, x_max, dx)
        self.n_cells = self.x.size + 1
        # Cell averaged-quantities
        # TODO: set initial conditions (same for 1D3V)
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
    def __init__(self, x_max, n_cells):
        self.x_max = x_max
        self.n_cells = n_cells
        self.x = np.linspace(0, x_max, n_cells, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        # External fields:
        self.E_0 = np.zeros((self.n_cells, 3))
        self.B_0 = np.zeros((self.n_cells, 3))
        # Total fields
        self.E = np.zeros((self.n_cells, 3))
        # the x component is always calculated directly using the poisson solver,
        # so it has to be added there each timestep
        self.E[:, 1:] = self.E_0[:, 1:]
        self.J = np.zeros((self.n_cells, 3))
        self.B = np.zeros((self.n_cells, 3)) + self.B_0

        # Cell averaged-quantities
        self.n_e = np.empty(self.n_cells)
        self.n_i = np.empty(self.n_cells)
        self.rho = np.empty(self.n_cells)

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        dummy = electrons.x / self.dx
        electrons.idx = dummy.astype(int)
        electrons.idx_staggered = (dummy - 0.5).astype(int)
        electrons.cic_weights = dummy - electrons.idx
        electrons.cic_weights_staggered = dummy - electrons.idx_staggered
        self.n_e.fill(0)
        np.add.at(self.n_e, electrons.idx, 1 - electrons.cic_weights)
        # TODO: We're assuming periodic BC here, take into account params.bc!
        np.add.at(self.n_e, (electrons.idx + 1) % self.n_cells, electrons.cic_weights)

        dummy = ions.x / self.dx
        ions.idx = dummy.astype(int)
        ions.idx_staggered = (dummy - 0.5).astype(int)
        ions.cic_weights = dummy - ions.idx
        ions.cic_weights_staggered = dummy - ions.idx_staggered
        self.n_i.fill(0)
        np.add.at(self.n_i, ions.idx, 1 - ions.cic_weights)
        np.add.at(self.n_i, (ions.idx + 1) % self.n_cells, ions.cic_weights)

        self.rho = electrons.q * self.n_e + ions.q * self.n_i
        # Remove mean charge, I'm not sure about the physical validity of doing this?
        # --> is only physically correct for quasi neutrality? so might not hold for open boundary at t!= 0.
        # Thus we might need to reconsider this for the 1D1V case as wel.
        # self.rho -= np.mean(self.rho)


# 2 spatial indices, 2/3 components
class Grid2D:
    # We assume that the y-axis is the same size as the x-axis
    def __init__(self, x_max, dx):
        self.x_max = x_max
        self.dx = dx
        self.x = np.arange(0, x_max, dx)
        self.y = np.arange(0, x_max, dx)
        self.n_cells = self.x.size + 1
        # The fields we consider are Ex, Ey and Bz
        # External fields:
        self.E_0 = np.zeros((self.n_cells, self.n_cells, 2))
        self.B_0 = np.zeros((self.n_cells, self.n_cells, 1))
        # Total fields
        self.E = np.zeros((self.n_cells, self.n_cells))
        # the x component is always calculated directly using the euler solver,
        # so it has to be added there each timestep
        # --> as we are choosing to not use the Euler solver for now we have to add the fields here
        self.E = np.zeros((self.n_cells, self.n_cells, 2)) + self.E_0
        self.J = np.zeros((self.n_cells, self.n_cells, 2))
        self.B = np.zeros((self.n_cells, self.n_cells, 1)) + self.B_0

        # Cell averaged-quantities
        self.n_e = np.empty((self.n_cells, self.n_cells))
        self.n_i = np.empty((self.n_cells, self.n_cells))
        self.rho = np.empty((self.n_cells, self.n_cells))

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        dummy = electrons.x / self.dx  # Subtract 1/2 * dx making sure that the two chosen cells are the closest after using astype
        electrons.idx = dummy.astype(int)
        electrons.cic_weights = dummy - electrons.idx
        self.n_e.fill(0)
        # TODO: We're assuming periodic BC here, take into account params.bc!
        # Create array to get the correct index for adjacent points
        x_adj = np.zeros((electrons.N, 2), dtype=int)
        y_adj = np.zeros((electrons.N, 2), dtype=int)
        x_adj[:, 0] = 1
        y_adj[:, 1] = 1
        np.add.at(self.n_e, (electrons.idx[:, 0], electrons.idx[:, 1]), (1 - electrons.cic_weights[:, 0]) * (1 - electrons.cic_weights[:, 1]))
        coords = (electrons.idx + x_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), electrons.cic_weights[:, 0] * (1 - electrons.cic_weights[:, 1]))
        coords = (electrons.idx + y_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), electrons.cic_weights[:, 1] * (1 - electrons.cic_weights[:, 0]))
        coords = (electrons.idx + x_adj + y_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), electrons.cic_weights[:, 0] * electrons.cic_weights[:, 1])

        dummy = ions.x / self.dx
        ions.idx = dummy.astype(int)
        ions.cic_weights = dummy - ions.idx
        self.n_i.fill(0)
        # TODO: We're assuming periodic BC here, take into account params.bc!
        # Create array to get the correct index for adjacent points
        x_adj = np.zeros((ions.N, 2), dtype=int)
        y_adj = np.zeros((ions.N, 2), dtype=int)
        x_adj[:, 0] = 1
        y_adj[:, 1] = 1
        np.add.at(self.n_e, (ions.idx[:, 0], ions.idx[:, 1]), (1 - ions.cic_weights[:, 0]) * (1 - ions.cic_weights[:, 1]))
        coords = (ions.idx + x_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), ions.cic_weights[:, 0] * (1 - ions.cic_weights[:, 1]))
        coords = (ions.idx + y_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), ions.cic_weights[:, 1] * (1 - ions.cic_weights[:, 0]))
        coords = (ions.idx + x_adj + y_adj) % self.n_cells
        np.add.at(self.n_e, (coords[:, 0], coords[:, 1]), ions.cic_weights[:, 0] * ions.cic_weights[:, 1])

        self.rho = electrons.q * self.n_e + ions.q * self.n_i
        # Remove mean charge, I'm not sure about the physical validity of doing this?
        # --> is only physically correct for quasi neutrality? so might not hold for open boundary at t!= 0.
        # Thus we might need to reconsider this for the 1D1V case as wel.
        self.rho -= np.mean(self.rho)
