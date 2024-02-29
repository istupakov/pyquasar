from typing import Callable, Optional
import numpy as np
from scipy import sparse
from scipy.integrate import quad_vec, fixed_quad
import numpy.typing as npt

from .fem import FemLine2


class BemLine2(FemLine2):
  """Represents a boundary element line element."""

  def __init__(
    self,
    element_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(element_verts, elements, quad_points, weights)
    self._basis_func = [
      lambda t: np.array([np.ones_like(t)]),
      lambda t: np.array([1 - t, t]),
      lambda t: np.array([1 - t**2, t**2]),
    ]
    self._basis_indices = [np.arange(len(elements), dtype=np.uint)[:, None], elements, elements]

  @property
  def basis_func(self) -> list[Callable]:
    """Basis functions of the boundary element line."""
    return self._basis_func

  @property
  def basis_indices(self) -> list[npt.NDArray[np.uint]]:
    """Indices of the boundary element line basis functions."""
    return self._basis_indices

  def potentials(
    self, points: npt.NDArray[np.floating]
  ) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
  ]:
    """Compute the potentials at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the potentials are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer and Newton potentials.
    """
    dr = points[..., None, :] - self.center
    lenghts = self.J.flatten()
    a = np.sum(self.dir * dr, axis=-1) / lenghts
    h = -np.sum(self.normal * dr, axis=-1)

    eps = 1e-30
    r0 = a**2 + h**2
    r1 = (lenghts - a) ** 2 + h**2
    log0 = np.log(r0 + eps)
    log1 = np.log(r1 + eps)
    atan0 = np.arctan(-a / (h + eps))
    atan1 = np.arctan((lenghts - a) / (h + eps))

    slpot = -((lenghts - a) * log1 + a * log0 + 2 * h * (atan1 - atan0) - 2 * lenghts) / (4 * np.pi)
    slpot_t = slpot * a / lenghts - (r1 * log1 - r0 * log0 + a**2 - (lenghts - a) ** 2) / (8 * np.pi) / lenghts
    dlpot = -(atan1 - atan0) / (2 * np.pi)
    dlpot[np.isclose(h, 0, atol=1e-10)] = 0
    dlpot_t = dlpot * a / lenghts - h * (log1 - log0) / (4 * np.pi) / lenghts
    nwpot = h * (lenghts / (8 * np.pi) + slpot / 2)

    return slpot, slpot_t, dlpot, dlpot_t, nwpot

  def mass_matrix(self, shape: tuple[int, ...], row_basis_order: int = 1, col_basis_order: int = 1) -> sparse.coo_array:
    """Compute the mass matrix of the boundary element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.
    row_basis_order : int
      The order of the basis functions for the rows.
    col_basis_order : int
      The order of the basis functions for the columns.

    Returns
    -------
    sparse.coo_array
    """
    row_basis = self.basis_func[row_basis_order](self.quad_points[:, 0])
    col_basis = self.basis_func[col_basis_order](self.quad_points[:, 0])
    data = self.J[:, None] * ((row_basis[None, :] * col_basis[:, None]) @ self.weights)
    i = np.broadcast_to(self.basis_indices[row_basis_order][:, None, :], data.shape)
    j = np.broadcast_to(self.basis_indices[col_basis_order][:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
    basis_order: int = 1,
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector of the boundary element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array to be integrated.
    shape : tuple[int, ...]
      The shape of the load vector.
    basis_order : int
      The order of the basis functions, by default 1.

    Returns
    -------
    NDArray[float]
    """
    basis = self.basis_func[basis_order](self.quad_points[:, 0])
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    data = self.J * ((basis * np.atleast_1d(f)[:, None]) @ self.weights)
    res = np.zeros(shape)
    np.add.at(res, self.basis_indices[basis_order], data)
    return res

  def bem_matrices(
    self, quad_order: Optional[int] = None
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line.

    Parameters
    ----------
    quad_order : int, optional
      The order of the quadrature, by default None. If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer V, double layer K and hypersingular D operators.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t * self.dir
      slpot, _, dlpot, dlpot_t, _ = self.potentials(r)
      dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray((slpot, dlpot_psi)), 1, -1)

    V, K = self.J * (quad_vec(f, 0, 1)[0][..., 0] if quad_order is None else fixed_quad(f, 0, 1, n=quad_order)[0])

    D = V / np.outer(self.J, self.J)
    D = np.take(-D, inv[0], axis=0) + np.take(D, inv[1], axis=0)
    D = np.take(-D, inv[0], axis=1) + np.take(D, inv[1], axis=1)

    return V, K, D

  def bem_matrices_p(self, order: Optional[int] = None) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line for the Neumann problem.

    Parameters
    ----------
    order : int, optional
      The order of the quadrature, by default None.
      If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float]]
      The single layer V and hypersingular D operators.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t * self.dir
      slpot, slpot_t, _, _, _ = self.potentials(r)
      slpot_psi = np.take(slpot - slpot_t, inv[0], axis=-1) + np.take(slpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray([(1 - t) * slpot_psi, t * slpot_psi, t * slpot_t]), 1, -1)

    pot = self.J * (quad_vec(f, 0, 1)[0][..., 0] if order is None else fixed_quad(f, 0, 1, n=order)[0])

    Vp = np.take(pot[0], inv[0], axis=0) + np.take(pot[1], inv[1], axis=0)

    Dp = pot[2] / np.outer(self.J, self.J)
    Dp = np.take(-Dp, inv[0], axis=0) + np.take(Dp, inv[1], axis=0)
    Dp = np.take(-Dp, inv[0], axis=1) + np.take(Dp, inv[1], axis=1)

    return Vp, Dp

  def result_weights(
    self, points: npt.NDArray[np.floating]
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the result weights at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the result weights are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer, and Newton potentials.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    slpot, _, dlpot, dlpot_t, nwpot = self.potentials(points)
    dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
    return slpot, dlpot_psi, np.sum(nwpot, axis=-1)

  def newton(self, points: npt.NDArray[np.floating], trace: int = 0) -> npt.NDArray[np.floating]:
    """Compute the Newton potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the Newton potential is evaluated.
    trace : int, optional
      The trace of the Newton potential, by default 0.

    Returns
    -------
    NDArray[float]
    """
    slpot, _, _, _, nwpot = self.potentials(points)
    return np.sum(nwpot, axis=-1) if trace == 0 else -np.sum(slpot[..., None] * self.normal, axis=-2)
