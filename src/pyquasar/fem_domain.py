from typing import Optional, Callable

import numpy as np
from scipy import sparse

from .fem import FemLine2, FemTriangle3
from .mesh import MeshDomain, MeshBlock
import numpy.typing as npt


class FemDomain:
  """Finite Element Method domain."""

  def __init__(self, domain: MeshDomain):
    self._mesh = domain
    self._element_count = sum(len(element.node_tags) for element in self.elements)

  @property
  def material(self) -> str:
    """Material of the FEM domain."""
    return self._mesh.material

  @property
  def boundary_indices(self) -> npt.NDArray[np.signedinteger]:
    """Indices of the FEM boundary nodes.

    Note
    ----
    The boundary indices are the global indices of the boundary nodes.
    """
    return self._mesh.boundary_indices

  @property
  def vertices(self) -> npt.NDArray[np.signedinteger]:
    """Vertices of the FEM domain."""
    return self._mesh.vertices

  @property
  def elements(self) -> list[MeshBlock]:
    """Finite elements of the FEM domain."""
    return self._mesh.elements

  @property
  def boundaries(self) -> list[MeshBlock]:
    """Boundary elements of the FEM domain."""
    return self._mesh.boundaries

  @property
  def dof_count(self) -> int:
    """Number of degrees of freedom."""
    return self._mesh.vertices.shape[0]

  @property
  def ext_dof_count(self) -> int:
    """Number of external degrees of freedom."""
    return self._mesh.boundary_indices.size

  @property
  def element_count(self) -> int:
    """Number of elements of FEM domain."""
    return self._element_count

  @property
  def stiffness_matrix(self) -> sparse.csc_matrix:
    """Global stiffness matrix of the FEM domain."""
    if hasattr(self, "_stiffness_matrix"):
      return self._stiffness_matrix
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def scaling(self) -> npt.NDArray[np.floating]:
    """Scaling of the domain."""
    if hasattr(self, "_scaling"):
      return self._scaling
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def kernel(self) -> npt.NDArray[np.floating]:
    """Kernel of basis."""
    if hasattr(self, "_kernel"):
      return self._kernel
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def load_vector(self) -> npt.NDArray[np.floating]:
    """Global load vector of the FEM domain."""
    if hasattr(self, "_load_vector"):
      return self._load_vector
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def corr_vector(self) -> npt.NDArray[np.floating]:
    """Global correction vector of the FEM domain."""
    if hasattr(self, "_corr_vector"):
      return self._corr_vector
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def diameter(self) -> tuple[float, float]:
    """Diameter of the FEM domain and its element."""
    if hasattr(self, "_diameter"):
      return self._diameter
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def neumann_factor(self) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Function for solving the factorized FEM Neumann problem."""
    if hasattr(self, "_neumann_factor"):
      return self._neumann_factor
    else:
      raise AttributeError("Domain is not decomposed yet.")

  @property
  def dirichlet_factor(self) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Function for solving the factorized FEM Dirichlet problem."""
    if hasattr(self, "_dirichlet_factor"):
      return self._dirichlet_factor
    else:
      raise AttributeError("Domain is not decomposed yet.")

  def __repr__(self) -> str:
    PAD = "\n\t"
    repr_str = f"<FemDomain object summary{PAD}DOF: {self.dof_count}{PAD}External DOF: {self.ext_dof_count}"
    if self.stiffness_matrix is not None:
      assembled = True
    else:
      assembled = False
    repr_str += f"{PAD}Assembled: {assembled}"
    if hasattr(self, "_neumann_factor"):
      decomposed = True
    else:
      decomposed = False
    repr_str += f"{PAD}Decomposed: {decomposed}"
    repr_str += f"{PAD}Mesh domain: {repr(self._mesh)}>"
    return repr_str

  def fabric(self, block: MeshBlock, ext: bool = False):
    """Return the corresponding FEM element.

    Parameters
    ----------
    block : MeshBlock
      The data containing the element tags, type, quadrature points and weights.
    ext : bool, optional
      Whether the boundary is external or not, by default False.

    Returns
    -------
    FemLine2 or FemTriangle3
      The corresponding FEM element.

    Raises
    ------
    ValueError
      If the element type is not supported.
    """
    indices = self.boundary_indices[block.node_tags] if ext else block.node_tags
    match block.type:
      case "Line 2":
        return FemLine2(self.vertices[block.node_tags], indices, block.quad_points, block.weights)
      case "Triangle 3":
        return FemTriangle3(self.vertices[block.node_tags], indices, block.quad_points, block.weights)
      case _:
        raise ValueError(f"Unsupported element type {block.type}")

  def assembly(
    self,
    material_dict: dict[
      Optional[str],
      Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    ],
  ) -> None:
    """Assemble the FEM domain. Stores the stiffness matrix in CSC format.

    Parameters
    ----------
    material_dict : dict[Optional[str], ArrayLike or Callable]
      The dictionary containing the materials.

    Raises
    ------
    ValueError
      If the element type is not supported.
    """
    self._load_vector = np.zeros(self.dof_count)
    self._corr_vector = np.zeros_like(self.load_vector)
    self._kernel = np.ones((1, self.load_vector.size))  # only for Lagrange basis
    lambda_ = material_dict.get("lambda", 1)

    for boundary in self.boundaries:
      if f := material_dict.get(boundary.type):
        for fe in map(self.fabric, boundary.elements):
          self._load_vector += np.sign(boundary.tag) * fe.load_vector(f, self.load_vector.shape)

    diameters = []
    self._scaling = np.full(self.ext_dof_count, lambda_)
    self._stiffness_matrix = sparse.coo_array((self.dof_count, self.dof_count))
    for fe in map(self.fabric, self.elements):
      self._stiffness_matrix += lambda_ * fe.stiffness_matrix(self._stiffness_matrix.shape)
      self._corr_vector += fe.load_vector(1, self.corr_vector.shape)
      diameters.append(fe.diameter())
      if f := material_dict.get(self.material):
        self._load_vector += fe.load_vector(f, self.load_vector.shape)
    self._diameter = ((sum_d := sum(D for D, _ in diameters)), sum(d * D for D, d in diameters) / sum_d)
    self._stiffness_matrix = self._stiffness_matrix.tocsc()

  def decompose(self) -> None:
    """Compute the factorization of the global matrix."""
    a = self._stiffness_matrix[0, 0]
    self._stiffness_matrix[0, 0] *= 2
    self._neumann_factor = sparse.linalg.factorized(self._stiffness_matrix)
    self._stiffness_matrix[0, 0] = a
    self._dirichlet_factor = sparse.linalg.factorized(self._stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :])

  def solve_neumann(self, flow) -> npt.NDArray[np.floating]:
    """Solve the FEM Neumann problem.

    Parameters
    ----------
    flow : Callable or ArrayLike
      The function or array-like object representing the flow.

    Returns
    -------
    NDArray[float]
    """

    def mult(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      return self.stiffness_matrix @ x + self.corr_vector * (self.corr_vector @ x)

    if hasattr(self, "_neumann_factor"):
      return self.neumann_factor(flow)
    return sparse.linalg.minres(sparse.linalg.LinearOperator(self.stiffness_matrix.shape, matvec=mult), flow, tol=1e-12)[0]

  def solve_dirichlet(self, disp: npt.NDArray[np.floating], lumped: bool = False) -> npt.NDArray[np.floating]:
    """Solve the FEM Dirichlet problem.

    Parameters
    ----------
    disp : NDArray[float]
      The displacement.
    lumped : bool, optional
      Whether to use lumped Dirichlet or not, by default False.

    Returns
    -------
    NDArray[float]
    """
    flow = self.stiffness_matrix[:, : self.ext_dof_count] @ disp[: self.ext_dof_count]
    if not lumped:
      if hasattr(self, "_dirichlet_factor"):
        sol = self.dirichlet_factor(flow[self.ext_dof_count :])
      else:
        sol = sparse.linalg.minres(
          self.stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :], flow[self.ext_dof_count :], tol=1e-12
        )[0]
      flow -= self.stiffness_matrix[:, self.ext_dof_count :] @ sol
    return flow

  def calc_solution(self, sol: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return sol
