from typing import Union

import grids
from particles import Particles


class Results:
    # NOTE: calling .append() is not super cheap, due to re-/deallocation of memory.
    #   However, I think that python internally doubles the array-length at each reallocation,
    #   which is definitely better than increasing by one on each .append() call.
    #   If we think about it, we might be able to remove some .copy() calls if arrays are replaced in the next iteration anyway
    #   (e.g. we do not have to copy grid.E if in the next iteration we do grid.E = "some new array",
    #    since this statement does not modify the previous array, but makes E point to "some new array")

    def __init__(self):
        # Keeps track of the time at which snapshots were made
        self.t = []
        # Keeps track of the different types of energy
        self.KE = []
        self.PE = []
        self.TE = []
        # Optionally, we could store the charge density as well, this is not done currently but is easily added
        self.n_e = []
        self.n_i = []
        # Keeps track of phase-space (x,v) of electrons and ions
        self.x_e = []
        self.v_e = []
        self.x_i = []
        self.v_i = []
        # Keeps track of the electromagnetic quantities (J and B will remain empty for 1D solver, while phi will only be non-empty for 1D solver)
        self.E = []
        self.phi = []
        self.J = []
        self.B = []
        # Keeps track of how cells shape the domain
        self.x = []

    def save_time(self, t: float):
        self.t.append(t)

    def save_energies(self, KE: float, PE: float, TE: float):
        self.KE.append(KE)
        self.PE.append(PE)
        self.TE.append(TE)

    def save_densities(self, grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D]):
        self.n_e.append(grid.n_e.copy())
        self.n_i.append(grid.n_i.copy())

    def save_phase_space(self, electrons: Particles, ions: Particles):
        self.x_e.append(electrons.x.copy())
        self.v_e.append(electrons.v.copy())
        self.x_i.append(ions.x.copy())
        self.v_i.append(ions.v.copy())

    def save_fields_1D(self, grid: grids.Grid1D):
        self.E.append(grid.E.copy())
        self.phi.append(grid.phi.copy())

    def save_fields_ND(self, grid: Union[grids.Grid1D3V, grids.Grid2D]):
        self.E.append(grid.E.copy())
        self.J.append(grid.J.copy())
        self.B.append(grid.B.copy())

    def save_cells(self, grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D]):
        self.x.append(grid.x.copy())
