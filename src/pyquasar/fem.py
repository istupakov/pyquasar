import numpy as np
from scipy import sparse

class FemBase:
  def __init__(self, elements, quad, weight):
    self.elements = elements
    self.quad = quad
    self.weight = weight

  def perp(self, vec):
    res = vec[..., ::-1].copy()
    np.negative(res[..., 0], out=res[..., 0])
    return res

  def vector(self, data, shape):
    res = np.zeros(shape)
    np.add.at(res, self.elements, data)
    return res

  def matrix(self, data, shape):
    i = np.broadcast_to(self.elements[:, None, :], data.shape)
    j = np.broadcast_to(self.elements[:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)

class FemBase1D(FemBase):
  def __init__(self, elem_vert, elements, quad, weight):
    super().__init__(elements, 0.5 * (1 + quad), 0.5 * weight);
    self.center = elem_vert[:, 0]
    self.dir = elem_vert[:, 1] - self.center
    self.J = np.linalg.norm(self.dir, axis=-1)[:, None]
    self.normal = -self.perp(self.dir) / self.J

  def diameter(self):
    return np.sum(self.J), np.mean(self.J)

class FemBase2D(FemBase):
  def __init__(self, elem_vert, elements, quad, weight):
    super().__init__(elements, quad, weight);
    self.center = elem_vert[:, 0]
    self.dir1 = elem_vert[:, 1] - self.center
    self.dir2 = elem_vert[:, 2] - self.center
    self.normal = np.cross(self.dir1, self.dir2).reshape(self.dir1.shape[0], -1)
    self.J = np.linalg.norm(self.normal, axis=-1)[:, None]
    self.normal /= self.J
    codir = [-self.perp(self.dir2), self.perp(self.dir1)] / self.J
    self.cometric = np.sum(codir[:, None, :] * codir[None, :, :], axis=-1)

  def diameter(self):
    return np.sum(self.J) ** 0.5, np.mean(self.J) ** 0.5

class FemLine2(FemBase1D):
  def mass_matrix(self, shape):
    psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
    return self.matrix(self.J[..., None] * ((psi[None, :] * psi[:, None]) @ self.weight), shape)

  def stiffness_matrix(self, shape):
    psiGrad = np.array([-1, 1])
    return self.matrix((psiGrad[None, :] * psiGrad[:, None]) / self.J[..., None], shape)

  def skew_grad_matrix(self, shape):
    psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
    psiGrad = np.array([-1, 1])[..., None]
    return self.matrix(np.ones_like(self.J[..., None]) * ((psi[None, :] * psiGrad[:, None]) @ self.weight), shape)

  def load_vector(self, func, shape):
    psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
    if callable(func):
      f = func(self.center[:, None] + self.quad[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = func
    return self.vector(self.J * ((psi * np.atleast_1d(f)[:, None]) @ self.weight), shape)

  def load_grad_vector(self, func, shape):
    psiGrad = np.array([-1, 1])
    if callable(func):
      f = func(self.center[:, None] + self.quad[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = func
    return self.vector((psiGrad * np.sum(np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weight, axis=-1)[:, None]) / self.J, shape)

class FemTriangle3(FemBase2D):
  def mass_matrix(self, shape):
    psi = np.array([1 - self.quad[:, 0] - self.quad[:, 1], self.quad[:, 0], self.quad[:, 1]])
    return self.matrix(self.J[..., None] * ((psi[None, :] * psi[:, None]) @ self.weight), shape)

  def stiffness_matrix(self, shape):
    psiGrad = np.array([[-1, 1, 0], [-1, 0, 1]])
    S = 0.5 * psiGrad[:, None, :, None] * psiGrad[None, :, None, :]
    return self.matrix(self.J[..., None] * np.sum(self.cometric[:, :, None, None] * S[..., None], axis=(0, 1)).T, shape)

  def load_vector(self, func, shape):
    psi = np.array([1 - self.quad[:, 0] - self.quad[:, 1], self.quad[:, 0], self.quad[:, 1]])
    if callable(func):
      point = self.center[:, None] + self.quad[None, :, 0, None] * self.dir1[:, None] + self.quad[None, :, 1, None] * self.dir2[:, None]
      f = func(point, self.normal[:, None])
    else:
      f = func
    return self.vector(self.J * ((psi * np.atleast_1d(f)[:, None]) @ self.weight), shape)
