from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from .bem import BemLine2
from .mesh import MeshDomain, MeshBlock


class BemDomain:
  """Boundary Element Method domain."""

  def __init__(self, domain: MeshDomain):
    self._mesh = domain

    bem_elements = []
    for boundary in self.boundaries:
      for block in boundary.elements:
        if boundary.tag > 0:
          bem_elements.append(block.node_tags)
        else:
          bem_elements.append(block.node_tags[..., ::-1])

    self._boundary_block = MeshBlock(
      self.boundaries[0].elements[0].type,
      np.concatenate(bem_elements),
      self.boundaries[0].elements[0].quad_points,
      self.boundaries[0].elements[0].weights,
    )
    self._element_count = len(self.boundary_block.node_tags)

  @property
  def material(self) -> str:
    """Material of the BEM domain."""
    return self._mesh.material

  @property
  def boundary_indices(self) -> npt.NDArray[np.signedinteger]:
    """Boundary indices of the BEM domain.

    Note
    ----
    The boundary indices are the global indices of the boundary nodes.
    """
    return self._mesh.boundary_indices

  @property
  def vertices(self) -> npt.NDArray[np.floating]:
    """Vertices of the BEM domain."""
    return self._mesh.vertices

  @property
  def elements(self) -> list[MeshBlock]:
    """Elements of the BEM domain."""
    return self._mesh.elements

  @property
  def boundaries(self) -> list[MeshBlock]:
    """Boundaries of the BEM domain."""
    return self._mesh.boundaries

  @property
  def dof_count(self) -> int:
    """Number of degrees of freedom."""
    return self._mesh.boundary_indices.size

  @property
  def ext_dof_count(self) -> int:
    """Number of external degrees of freedom."""
    return self._mesh.boundary_indices.size

  @property
  def boundary_block(self) -> MeshBlock:
    """Boundary block of the BEM domain."""
    return self._boundary_block

  @property
  def element_count(self) -> int:
    """Number of elements in BEM domain."""
    return self._element_count

  @property
  def load_vector(self) -> npt.NDArray[np.floating]:
    """Load vector of the BEM domain."""
    if hasattr(self, "_load_vector"):
      return self._load_vector
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def kernel(self) -> npt.NDArray[np.floating]:
    """Kernel of the BEM domain."""
    if hasattr(self, "_kernel"):
      return self._kernel
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def scaling(self) -> npt.NDArray[np.floating]:
    """Scaling of the BEM domain."""
    if hasattr(self, "_scaling"):
      return self._scaling
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def diameter(self) -> tuple[float, float]:
    """Diameter of the BEM domain and its element."""
    if hasattr(self, "_diameter"):
      return self._diameter
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def S(self) -> npt.NDArray[np.floating]:
    """Poincaré–Steklov S operator of the BEM domain."""
    if hasattr(self, "_S"):
      return self._S
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def D(self) -> npt.NDArray[np.floating]:
    """Hypersingular boundary integral operator D operator of the BEM domain."""
    if hasattr(self, "_D"):
      return self._D
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def inv_S(self) -> npt.NDArray[np.floating]:
    """Inverse of the Poincaré–Steklov S operator of the BEM domain."""
    if hasattr(self, "_inv_S"):
      return self._inv_S
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def inv_VK(self) -> npt.NDArray[np.floating]:
    """Inverse of the product of single layer operator V and double layer operator K of the BEM domain."""
    if hasattr(self, "_inv_VK"):
      return self._inv_VK
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def inv_VN(self) -> npt.NDArray[np.floating]:
    """Inverse of the of single layer operator V and operator N of the BEM domain."""
    if hasattr(self, "_inv_VN"):
      return self._inv_VN
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def f(self) -> npt.NDArray[np.floating]:
    """Material function of the BEM domain."""
    if hasattr(self, "_f"):
      return self._f
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def vertices_weights(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Weights of the vertices of the BEM domain."""
    if hasattr(self, "_vertices_weights"):
      return self._vertices_weights
    else:
      raise AttributeError("Domain is not assembled yet.")

  def fabric(self, block: MeshBlock, ext: bool = False) -> BemLine2:
    """Return the corresponding BEM element.

    Parameters
    ----------
    block : MeshBlock
      The data containing the element tags, type, quadrature points and weights.
    ext : bool, optional
      Whether the boundary is external or not, by default False.

    Returns
    -------
    BemLine2
      The corresponding BEM element.

    Raises
    ------
    ValueError
      If the element type is not supported.
    """
    indices = self.boundary_indices[block.node_tags] if ext else block.node_tags
    match block.type:
      case "Line 2":
        return BemLine2(self.vertices[block.node_tags], indices, block.quad_points, block.weights)
      case _:
        raise ValueError(f"Unsupported element type {block.type}")

  def assembly(
    self,
    material_dict: dict[
      Optional[str], Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike
    ],
  ) -> None:
    """Assemble the BEM domain.

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
    self._kernel = np.ones((1, self.load_vector.size))
    k = material_dict.get("lambda", 1)
    self._scaling = np.full(self.ext_dof_count, k)

    for boundary in self.boundaries:
      if f := material_dict.get(boundary.type):
        for be in map(self.fabric, boundary.elements):
          self._load_vector += np.sign(boundary.tag) * be.load_vector(f, self.load_vector.shape)

    be = self.fabric(self.boundary_block)
    self._diameter = be.diameter()
    V, K, self._D = be.bem_matrices(5)
    M = be.mass_matrix(K.shape, 0, 1)
    K += M / 2

    inv_V = np.linalg.inv(V)
    self._inv_VK = inv_V @ K

    corr0 = be.load_vector(1, V.shape[0], 0)
    w = inv_V @ corr0
    alpha = k / (4 * (w @ corr0))
    corr_vector = M @ w

    self._S = k * (self.D + K.T @ self.inv_VK)
    self._D *= k
    self._inv_S = np.linalg.inv(self.S + alpha * np.outer(corr_vector, corr_vector))

    if f := material_dict.get(self.material):
      assert np.isscalar(f) and np.isreal(f), "Material function must be a real scalar"
      N0 = f * be.load_vector(lambda p, n: be.newton(p, 0), V.shape[0], 0)
      N1 = f * be.load_vector(lambda p, n: (t := be.newton(p, 1))[..., 0] * n[..., 0] + t[..., 1] * n[..., 1], self.load_vector.shape, 1)
      self._inv_VN = inv_V @ N0
      self._load_vector += K.T @ self.inv_VN - N1
      self._f = f
    else:
      self._inv_VN = 0
      self._f = 0

    self._vertices_weights = be.result_weights(self.vertices)

  def decompose(self) -> None:
    """Empty method for API compatibility with FETI problem."""
    pass

  def solve_neumann(self, flow: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Solve the BEM Neumann problem.

    Parameters
    ----------
    flow : Callable or ArrayLike
      The function or array-like object representing the flow.

    Returns
    -------
    NDArray[float]
    """
    return self.inv_S @ flow

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
    if lumped:
      return self.D @ disp
    else:
      return self.S @ disp

  def calc_solution(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Calculate the problem solution.

    Parameters
    ----------
    q : NDArray[float]
      The coefficients of discretization of a function over a basis.

    Returns
    -------
    NDArray[float]
    """
    p = self.inv_VK @ q - self.inv_VN
    res = self.vertices_weights[0] @ p - self.vertices_weights[1] @ q + self.vertices_weights[2] * self.f
    res[: q.size] = q
    return res
