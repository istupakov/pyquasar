import numpy as np
from scipy import sparse
from scipy.integrate import quad_vec, fixed_quad

from .fem import FemLine2

class BemLine2(FemLine2):
  def __init__(self, elem_vert, elements, quad, weight):
    super().__init__(elem_vert, elements, quad, weight);
    self.basis_func = [lambda t: np.array([np.ones_like(t)]), lambda t: np.array([1-t, t]), lambda t: np.array([1-t**2, t**2])]
    self.basis_indices = [np.arange(len(elements))[:, None], elements, elements]

  def potentials(self, points):
    dr = points[..., None, :] - self.center
    l = self.J.flatten()
    a = np.sum(self.dir * dr, axis=-1)/l
    h = -np.sum(self.normal * dr, axis=-1)

    eps = 1e-30
    r0 = a**2 + h**2
    r1 = (l-a)**2 + h**2
    log0 = np.log(r0 + eps)
    log1 = np.log(r1 + eps)
    atan0 = np.arctan(-a/(h + eps))
    atan1 = np.arctan((l - a)/(h + eps))

    slpot = -((l - a)*log1 + a*log0 + 2*h*(atan1 - atan0) - 2*l)/(4*np.pi)
    slpot_t = slpot*a/l - (r1*log1 - r0*log0 + a**2 - (l-a)**2)/(8*np.pi)/l
    dlpot = -(atan1 - atan0)/(2*np.pi)
    dlpot[np.isclose(h, 0, atol=1e-10)] = 0
    dlpot_t = dlpot*a/l - h*(log1 - log0)/(4*np.pi)/l
    nwpot = h*(l/(8*np.pi) + slpot/2)

    return slpot, slpot_t, dlpot, dlpot_t, nwpot

  def mass_matrix(self, shape, row_basis_order = 1, col_basis_order = 1):
    row_basis = self.basis_func[row_basis_order](self.quad[:, 0])
    col_basis = self.basis_func[col_basis_order](self.quad[:, 0])
    data = self.J[:, None] * ((row_basis[None, :] * col_basis[:, None]) @ self.weight)
    i = np.broadcast_to(self.basis_indices[row_basis_order][:, None, :], data.shape)
    j = np.broadcast_to(self.basis_indices[col_basis_order][:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)

  def load_vector(self, func, shape, basis_order = 1):
    basis = self.basis_func[basis_order](self.quad[:, 0])
    if callable(func):
      f = func(self.center[:, None] + self.quad[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
      # def f(t):
      #   basis = self.basis_func[basis_order](t)
      #   f = func(self.center[:, None] + t[None, :, None] * self.dir[:, None], self.normal[:, None])
      #   return basis * f[:, None]
      # data = self.J * fixed_quad(f, 0, 1, n = 10)[0]
    else:
      f = func
    data = self.J * ((basis * np.atleast_1d(f)[:, None]) @ self.weight)
    res = np.zeros(shape)
    np.add.at(res, self.basis_indices[basis_order], data)
    return res

  def bem_matrices(self, quadrature_order = None):
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t):
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t*self.dir
      slpot, slpot_t, dlpot, dlpot_t, nwpot = self.potentials(r)
      dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
      return np.moveaxis((slpot, dlpot_psi), 1, -1)

    V, K = self.J*(quad_vec(f, 0, 1)[0][..., 0] if quadrature_order is None else fixed_quad(f, 0, 1, n = quadrature_order)[0])

    D = V/np.outer(self.J, self.J)
    D = np.take(-D, inv[0], axis=0) + np.take(D, inv[1], axis=0)
    D = np.take(-D, inv[0], axis=1) + np.take(D, inv[1], axis=1)

    return V, K, D

  def bem_matrices_p(self, order = None):
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t):
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t*self.dir
      slpot, slpot_t, dlpot, dlpot_t, nwpot = self.potentials(r)
      slpot_psi = np.take(slpot - slpot_t, inv[0], axis=-1) + np.take(slpot_t, inv[1], axis=-1)
      return np.moveaxis([(1-t)*slpot_psi, t*slpot_psi, t * slpot_t], 1, -1)

    pot = self.J*(quad_vec(f, 0, 1)[0][..., 0] if order is None else fixed_quad(f, 0, 1, n = order)[0])

    Vp = np.take(pot[0], inv[0], axis=0) + np.take(pot[1], inv[1], axis=0)

    Dp = pot[2]/np.outer(self.J, self.J)
    Dp = np.take(-Dp, inv[0], axis=0) + np.take(Dp, inv[1], axis=0)
    Dp = np.take(-Dp, inv[0], axis=1) + np.take(Dp, inv[1], axis=1)

    return Vp, Dp

  def result_weights(self, points):
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    slpot, slpot_t, dlpot, dlpot_t, nwpot = self.potentials(points)
    dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
    return slpot, dlpot_psi, np.sum(nwpot, axis=-1)

  def newton(self, points, trace = 0):
    slpot, slpot_t, dlpot, dlpot_t, nwpot = self.potentials(points)
    return np.sum(nwpot, axis=-1) if trace == 0 else -np.sum(slpot[..., None] * self.normal, axis=-2)
