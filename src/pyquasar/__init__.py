from .load_mesh import load_mesh
from .fem_domain import FemDomain
from .bem_domain import BemDomain
from .feti_problem import FetiProblem, FetiProblemNotRed
from .coils import Coil2D

__all__ = [
    "BemDomain",
    "Coil2D",
    "FemDomain",
    "FetiProblem",
    "FetiProblemNotRed",
    "load_mesh",
]
