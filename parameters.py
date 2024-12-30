from dataclasses import dataclass
from enum import Enum


class BoundaryCondition(Enum):
    Open = 0
    Absorbing = 1
    Periodic = 2


@dataclass
class Parameters:
    x_max: float
    n_cells: int
    t_max: float
    max_iter: int
    bc: BoundaryCondition
    dimX: int
    dimV: int
    num_particles: int = -1
    damping_width: float = 0
    SOR_max_iter: int = 1000
    SOR_tol: float = 1.0e-6
    SOR_omega: float = 1.5
    seed: int = 42
    dx: float = 0.0

    def __post_init__(self):
        self.dx = self.x_max / self.n_cells

    def __repr__(self):
        # Generate a string representation with one field per line
        field_strings = [f"{field}: {getattr(self, field)}" for field in self.__dataclass_fields__]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(field_strings) + "\n)"

    __str__ = __repr__
