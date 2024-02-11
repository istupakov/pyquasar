import numpy as np
from scipy import sparse

from .fem import FemLine2, FemTriangle3

class FemDomain:
  def __init__(self, material, boundary_indices, vertices, elements, boundaries):
    self.material = material
    self.boundary_indices = boundary_indices
    self.vertices = vertices
    self.elements = elements
    self.boundaries = boundaries
    self.dof_count = self.vertices.shape[0]
    self.ext_dof_count = self.boundary_indices.size
    self.element_count = sum(len(e) for _, e, _, _ in self.elements)

  def fabric(self, data, ext = False):
    indices = self.boundary_indices[data[1]] if ext else data[1]
    match data[0]:
      case "Line 2":
        return FemLine2(self.vertices[data[1]], indices, *data[2:])
      case "Triangle 3":
        return FemTriangle3(self.vertices[data[1]], indices, *data[2:])
      case _:
        raise Exception(f"Unsupported element type {data[0]}")

  def assembly(self, material_dict):
    self.load_vector = np.zeros(self.dof_count)
    self.corr_vector = np.zeros_like(self.load_vector)
    self.kernel = np.ones((1, self.load_vector.size)) # only for Lagrange basis
    k = material_dict.get('coeff', 1)

    for bmat, tag, elements in self.boundaries:
      if f := material_dict.get(bmat):
        for fem in map(self.fabric, elements):
          self.load_vector += np.sign(tag) * fem.load_vector(f, self.load_vector.shape)

    diameters = []
    self.scaling = np.full(self.ext_dof_count, k)
    self.stiffness_matrix = sparse.coo_array((self.dof_count, self.dof_count))
    for fem in map(self.fabric, self.elements):
      self.stiffness_matrix += k * fem.stiffness_matrix(self.stiffness_matrix.shape)
      self.corr_vector += fem.load_vector(1, self.corr_vector.shape)
      diameters.append(fem.diameter())
      if f := material_dict.get(self.material):
        self.load_vector += fem.load_vector(f, self.load_vector.shape)
    self.diameter = ((sumD := sum(D for D, d in diameters)), sum(d * D for D, d in diameters)/sumD)

    self.stiffness_matrix = self.stiffness_matrix.tocsc()#tocsr()
    self.neumann_factor = None
    self.dirichlet_factor = None

  def decompose(self):
    a = self.stiffness_matrix[0, 0]
    self.stiffness_matrix[0, 0] *= 2
    self.neumann_factor = sparse.linalg.factorized(self.stiffness_matrix)
    self.stiffness_matrix[0, 0] = a
    self.dirichlet_factor = sparse.linalg.factorized(self.stiffness_matrix[self.ext_dof_count:, self.ext_dof_count:])

  def solve_neumann(self, flow):
    def mult(x):
      return self.stiffness_matrix @ x + self.corr_vector * (self.corr_vector @ x)
    if self.neumann_factor:
      return self.neumann_factor(flow)
    return sparse.linalg.minres(sparse.linalg.LinearOperator(self.stiffness_matrix.shape, matvec=mult), flow, tol=1e-12)[0]

  def solve_dirichlet(self, disp, lumped = False):
    flow = self.stiffness_matrix[:, :self.ext_dof_count] @ disp[:self.ext_dof_count]
    if not lumped:
      if self.dirichlet_factor:
        sol = self.dirichlet_factor(flow[self.ext_dof_count:])
      else:
        sol = sparse.linalg.minres(self.stiffness_matrix[self.ext_dof_count:, self.ext_dof_count:], flow[self.ext_dof_count:], tol=1e-12)[0]
      flow -= self.stiffness_matrix[:, self.ext_dof_count:] @ sol
    return flow

  def calc_solution(self, sol):
    return sol
