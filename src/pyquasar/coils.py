import numpy as np
import numpy.typing as npt
from .bem import BemLine2


class Coil2D:
  """
  Represents a 2D coil.

  Methods
  -------
  calc_A(points: NDArray[float]) -> NDArray[float]:
    Calculates the vector potential at the given points.

  calc_grad_A(NDArray[float]) -> NDArray[float]
    Calculates the gradient of the vector potential at the given points.

  calc_rot_A(points: NDArray[float]) -> NDArray[float]:
    Calculates the curl of the vector potential at the given points.

  """

  def __init__(self, vertices: npt.NDArray[np.floating]):
    vertices = np.asarray(vertices)
    indices = np.arange(vertices[..., 0].size).reshape(vertices[..., 0].shape)
    elements = np.dstack((indices, np.roll(indices, -1, axis=-1))).reshape(-1, 2)
    vertices = vertices.reshape(-1, 2)
    self.bem = BemLine2(vertices[elements], elements, np.asarray([[0]]), np.asarray([[0]]))

  def calc_A(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Calculates the vector potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      An array-like object containing the points at which to calculate the vector potential.

    Returns
    -------
    NDArray[float]
    """
    return self.bem.newton(points)

  def calc_grad_A(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Calculates the gradient of the vector potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      An array-like object containing the points at which to calculate the gradient.

    Returns:
    -------
    NDArray[float]
    """
    return self.bem.newton(points, 1)

  def calc_rot_A(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Calculates the curl of the vector potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      An array-like object containing the points at which to calculate the curl.

    Returns:
    -------
    NDArray[float]
    """
    return -self.bem.perp(self.bem.newton(points, 1))
