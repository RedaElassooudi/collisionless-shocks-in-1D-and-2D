from dataclasses import dataclass
from enum import Enum


class BoundaryCondition(Enum):
    Open = 0
    Absorbing = 1
    Periodic = 2


@dataclass
class Parameters:
    x_max: float
    dx: float
    t_max: float
    max_iter: int
    bc: BoundaryCondition
    damping_width: float = 0
    SOR_max_iter: int = 1000
    SOR_tol: float = 1.0e-6
    SOR_omega: float = 1.5
    seed: int = 42
