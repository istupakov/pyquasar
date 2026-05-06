"""Finite and boundary element methods for two-dimensional model problems."""

from .bem_domain import BemDomain
from .coils import Coil2D
from .fem_domain import FemDomain
from .feti_problem import FetiProblem, FetiProblemNotRed
from .load_mesh import load_mesh

__all__ = [
    "BemDomain",
    "Coil2D",
    "FemDomain",
    "FetiProblem",
    "FetiProblemNotRed",
    "load_mesh",
]
