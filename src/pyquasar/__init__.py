from .fem_domain import FemDomain
from .bem_domain import BemDomain
from .feti_problem import FetiProblem, FetiProblemNotRed
from .coils import Coil2D
from .mesh import Mesh

__all__ = ["FemDomain", "BemDomain", "FetiProblem", "FetiProblemNotRed", "Coil2D", "Mesh"]
