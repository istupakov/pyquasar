import numpy as np

from .bem import BemLine2

class BemDomain:
  def __init__(self, material, boundary_indices, vertices, elements, boundaries):
    self.material = material
    self.boundary_indices = boundary_indices
    self.vertices = vertices
    self.elements = elements
    self.boundaries = boundaries
    self.dof_count = self.boundary_indices.size
    self.ext_dof_count = self.boundary_indices.size

    bem_elements = [block[1] if tag > 0 else block[1][..., ::-1] for bmat, tag, elements in self.boundaries for block in elements]
    self.bem_elements = (self.boundaries[0][2][0][0], np.concatenate(bem_elements), self.boundaries[0][2][0][2], self.boundaries[0][2][0][3])
    self.element_count = len(self.bem_elements[1])

  def fabric(self, data, ext = False):
    indices = self.boundary_indices[data[1]] if ext else data[1]
    match data[0]:
      case "Line 2":
        return BemLine2(self.vertices[data[1]], indices, *data[2:])
      case _:
        raise Exception(f"Unsupported element type {data[0]}")

  def assembly(self, material_dict):
    self.load_vector = np.zeros(self.dof_count)
    self.kernel = np.ones((1, self.load_vector.size))
    k = material_dict.get('coeff', 1)
    self.scaling = np.full(self.ext_dof_count, k)

    for bmat, tag, elements in self.boundaries:
      if f := material_dict.get(bmat):
        for fem in map(self.fabric, elements):
          self.load_vector += np.sign(tag) * fem.load_vector(f, self.load_vector.shape)

    bem = self.fabric(self.bem_elements)
    self.diameter = bem.diameter()
    V, K, self.D = bem.bem_matrices(5)
    M = bem.mass_matrix(K.shape, 0, 1)
    K += M/2

    invV = np.linalg.inv(V)
    self.invVK = invV @ K

    corr0 = bem.load_vector(1, V.shape[0], 0)
    w = invV @ corr0
    alpha = k/(4 * (w @ corr0))
    corr_vector = M @ w

    self.S = k * (self.D + K.T @ self.invVK)
    self.D *= k
    self.invS = np.linalg.inv(self.S + alpha * np.outer(corr_vector, corr_vector))

    if f := material_dict.get(self.material):
      assert np.isscalar(f) and np.isreal(f)
      N0 = f * bem.load_vector(lambda p, n: bem.newton(p, 0), V.shape[0], 0)
      N1 = f * bem.load_vector(lambda p, n: (t := bem.newton(p, 1))[..., 0] * n[..., 0] + t[..., 1] * n[..., 1], self.load_vector.shape, 1)
      self.invVN = invV @ N0
      self.load_vector += K.T @ self.invVN - N1
      self.f = f
    else:
      self.invVN = 0
      self.f = 0

    self.vertices_weights = bem.result_weights(self.vertices)

  def decompose(self): pass

  def solve_neumann(self, flow):
    return self.invS @ flow

  def solve_dirichlet(self, disp, lumped = False):
    if lumped:
      return self.D @ disp
    else:
      return self.S @ disp

  def calc_solution(self, q):
    p = self.invVK @ q - self.invVN
    res = self.vertices_weights[0] @ p - self.vertices_weights[1] @ q + self.vertices_weights[2] * self.f
    res[:q.size] = q
    return res
