import numpy as np

from .bem import BemLine2

class Coil2D:
  def __init__(self, vertices):
    vertices = np.asarray(vertices)
    indices = np.arange(vertices[..., 0].size).reshape(vertices[..., 0].shape)
    elements = np.dstack((indices, np.roll(indices, -1, axis=-1))).reshape(-1, 2)
    vertices = vertices.reshape(-1, 2)
    self.bem = BemLine2(vertices[elements], elements, 0, 0)

  def calcA(self, points):
    return self.bem.newton(points)

  def calc_gradA(self, points):
    return self.bem.newton(points, 1)

  def calc_rotA(self, points):
    return -self.bem.perp(self.bem.newton(points, 1))
