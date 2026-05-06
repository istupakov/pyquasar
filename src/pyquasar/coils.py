"""Coil post-processing helpers built on boundary elements."""

import numpy as np
from numpy.typing import ArrayLike

from ._typing import FloatArray
from .bem import BemLine2


class Coil2D:
    """Closed two-dimensional coil represented by boundary line elements."""

    def __init__(self, vertices: ArrayLike) -> None:
        """Create a coil from ordered polygon vertices.

        Parameters
        ----------
        vertices
            Array-like collection of two-dimensional vertices. The last vertex
            is connected back to the first one automatically.
        """
        vertices = np.asarray(vertices)
        indices = np.arange(vertices[..., 0].size).reshape(vertices[..., 0].shape)
        elements = np.dstack((indices, np.roll(indices, -1, axis=-1))).reshape(-1, 2)
        vertices = vertices.reshape(-1, 2)
        self.bem = BemLine2(vertices[elements], elements, 0, 0)

    def calcA(self, points: ArrayLike) -> FloatArray:
        """Evaluate the magnetic vector potential at points."""
        return self.bem.newton(np.asarray(points))

    def calc_gradA(self, points: ArrayLike) -> FloatArray:
        """Evaluate the gradient of the magnetic vector potential at points."""
        return self.bem.newton(np.asarray(points), 1)

    def calc_rotA(self, points: ArrayLike) -> FloatArray:
        """Evaluate the rotated vector-potential gradient at points."""
        return -self.bem.perp(self.bem.newton(np.asarray(points), 1))
