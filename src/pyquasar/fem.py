from typing import Callable
import numpy as np
from scipy import sparse
import numpy.typing as npt


class FemBase:
  """A base class for Finite Element Method."""

  def __init__(
    self,
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    self._elements = elements
    self._quad_points = quad_points
    self._weights = weights

  @property
  def elements(self) -> npt.NDArray[np.signedinteger]:
    """The indices of elements."""
    return self._elements

  @property
  def quad_points(self) -> npt.NDArray[np.floating]:
    """The points of quadrature."""
    return self._quad_points

  @property
  def weights(self) -> npt.NDArray[np.floating]:
    """The weights of quadrature."""
    return self._weights

  def perp(self, vec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the perpendicular vector to the input vector.

    Parameters
    ----------
    vec : NDArray[float]
      The input vector.

    Returns
    -------
    NDArray[float]
      The perpendicular vector to the input vector.
    """
    res = vec[..., ::-1].copy()
    np.negative(res[..., 0], out=res[..., 0])
    return res

  def vector(self, data: npt.NDArray[np.floating], shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
    """Transform local vectors to global vector.

    Parameters
    ----------
    data : NDArray[float]
      The local vectors.
    shape : tuple[int, ...]
      The shape of the global vector.

    Returns
    -------
    NDArray[float]
      The global vector.
    """
    res = np.zeros(shape)
    np.add.at(res, self.elements, data)
    return res

  def matrix(self, data: npt.NDArray[np.floating], shape: tuple[int, ...]) -> sparse.coo_array:
    """Construct a sparse COO matrix from the local matrices and given shape.

    Parameters
    ----------
    data : NDArray[float]
      The data array containing the local matrices.
    shape : tuple[int, ...]
      The shape of the matrix.

    Returns
    -------
    coo_array
      The constructed global sparse COO matrix.
    """
    i = np.broadcast_to(self.elements[:, None, :], data.shape)
    j = np.broadcast_to(self.elements[:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)


class FemBase1D(FemBase):
  """A base class for 1D Finite Element Method."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements, 0.5 * (1 + quad_points), 0.5 * weights)
    self._center = elements_verts[:, 0]
    self._dir = elements_verts[:, 1] - self.center
    self._J = np.linalg.norm(self.dir, axis=-1)[:, None]
    self._normal = -self.perp(self.dir) / self.J

  @property
  def center(self) -> npt.NDArray[np.floating]:
    """The center of the element vertices."""
    return self._center

  @property
  def dir(self) -> npt.NDArray[np.floating]:
    """The direction of the element vertices."""
    return self._dir

  @property
  def J(self) -> npt.NDArray[np.floating]:
    """The Jacobian of elements."""
    return self._J

  @property
  def normal(self) -> npt.NDArray[np.floating]:
    """The normal of the direction."""
    return self._normal

  def diameter(self) -> tuple[np.floating, np.floating]:
    """Returns domain diameter and its element."""
    return np.sum(self.J), np.mean(self.J)


class FemBase2D(FemBase):
  """A base class for 2D Finite Element Method."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements, quad_points, weights)
    self._center = elements_verts[:, 0]
    self._dir1 = elements_verts[:, 1] - self.center
    self._dir2 = elements_verts[:, 2] - self.center
    self._normal = np.cross(self.dir1, self.dir2).reshape(self.dir1.shape[0], -1)
    self._J = np.linalg.norm(self.normal, axis=-1)[:, None]
    self._normal /= self.J
    self._contradir = [-self.perp(self.dir2), self.perp(self.dir1)] / self.J
    self._contrametric = np.sum(self.contradir[:, None, :] * self.contradir[None, :, :], axis=-1)

  @property
  def center(self) -> npt.NDArray[np.floating]:
    """The center of the element vertices."""
    return self._center

  @property
  def dir1(self) -> npt.NDArray[np.floating]:
    """The first direction of the element vertices."""
    return self._dir1

  @property
  def dir2(self) -> npt.NDArray[np.floating]:
    """The second direction of the element vertices."""
    return self._dir2

  @property
  def normal(self) -> npt.NDArray[np.floating]:
    """The normal of the element vertices."""
    return self._normal

  @property
  def J(self) -> npt.NDArray[np.floating]:
    """The Jacobian of the elements."""
    return self._J

  @property
  def contradir(self) -> npt.NDArray[np.floating]:
    """The contravariant direction of the elements."""
    return self._contradir

  @property
  def contrametric(self) -> npt.NDArray[np.floating]:
    """The contravariant metric of the elements."""
    return self._contrametric

  def diameter(self) -> tuple[np.floating, np.floating]:
    """Returns domain diameter and its element."""
    return np.sum(self.J) ** 0.5, np.mean(self.J) ** 0.5


class FemLine2(FemBase1D):
  """Represents a finite element line in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array([1 - self.quad_points[:, 0], self.quad_points[:, 0]])
    self._psi_grad = np.array([-1, 1])

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element line."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element line."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix((self.psi_grad[None, :] * self.psi_grad[:, None]) / self.J[..., None], shape)

  def skew_grad_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the skew gradient matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the skew gradient matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(np.ones_like(self.J[..., None]) * ((self.psi[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def load_grad_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load gradient vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(
      (self.psi_grad * np.sum(np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weights, axis=-1)[:, None]) / self.J,
      shape,
    )


class FemTriangle3(FemBase2D):
  """Represents a finite element triangle with 3 nodes in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array([1 - self.quad_points[:, 0] - self.quad_points[:, 1], self.quad_points[:, 0], self.quad_points[:, 1]])
    self._psi_grad = np.array([[-1, 1, 0], [-1, 0, 1]])

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element triangle."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element triangle."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
    """Compute the mass matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
    """Compute the stiffness matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    S = 0.5 * self.psi_grad[:, None, :, None] * self.psi_grad[None, :, None, :]
    return self.matrix(self.J[..., None] * np.sum(self.contrametric[:, :, None, None] * S[..., None], axis=(0, 1)).T, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element triangle.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
      )
      f = func(point, self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)
