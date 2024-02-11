import numpy as np
from scipy import sparse

class FetiProblem:
  def __init__(self, domains, dim = 2):
    self.domains = domains
    self.dim = domains[0].vertices.shape[-1]

  def build_constraints(self, links, constraints):
    idx = np.nonzero(links >= 0)[0]
    if idx[0] == 0:
      for j in idx[1:]:
        constraints[j - 1].append((self.dual_size, links[j], +1))
        self.dual_size += 1
      return

    for i in idx:
      for j in idx:
        if i < j:
          constraints[i - 1].append((self.dual_size, links[i], +1))
          constraints[j - 1].append((self.dual_size, links[j], -1))
          self.dual_size += 1

  def assembly_constraints(self, dirichlet_name):
    max_index = max(domain.boundary_indices.max() for domain in self.domains)
    connections = np.full((max_index + 1, len(self.domains) + 1), -1, dtype=int)

    for i in range(len(self.domains)):
      dir_internal_indices = [block[1] for bmat, _, elements in self.domains[i].boundaries for block in elements if bmat == dirichlet_name]
      dirichlet = self.domains[i].boundary_indices[np.unique(np.concatenate(dir_internal_indices))] if len(dir_internal_indices) else []
      connections[dirichlet, 0] = dirichlet
      boundary = self.domains[i].boundary_indices
      connections[boundary, i + 1] = np.arange(boundary.size)
    self.boundary_types = np.sum(connections >= 0, axis=-1) > 2
    self.dirichlet_boundary = connections[:, 0] >= 0

    constraints = [[] for i in range(connections.shape[1] - 1)]
    self.dual_size = 0
    for links in connections:
      self.build_constraints(links, constraints)

    self.B = []
    for constraint, domain in zip(constraints, self.domains):
      i, j, v = np.array(constraint).T
      self.B.append(sparse.coo_array((v, (i, j)), (self.dual_size, domain.dof_count)).tocsr())

  def condition_number_estimate(self):
    return np.log1p(max(d.diameter[0]/d.diameter[1] for d in self.domains)) ** 2

  def assembly_scaling(self):
    jumps = np.full((self.dual_size, 2), np.inf)
    q = np.full(self.dual_size, np.inf)
    scaling = np.zeros(self.boundary_types.shape)
    Bcoo = [B.tocoo() for B in self.B]
    for domain, B in zip(self.domains, Bcoo):
      scaling[domain.boundary_indices] += domain.scaling
      jumps[B.row, 1 * (B.data == -1)] = domain.scaling[B.col]
      H, h = domain.diameter
      q[B.row] = np.min((q[B.row], np.where(self.boundary_types[domain.boundary_indices[B.col]], h**(self.dim-2), (1 + np.log(H/h)*h**(self.dim-1)/H))), axis=0)

    Q = sparse.diags(q * np.min(jumps, axis=-1))
    Bs = []
    for domain, B in zip(self.domains, Bcoo):
      val = jumps[B.row, 1 * (B.data == 1)] / scaling[domain.boundary_indices[B.col]]
      Bs.append(sparse.coo_array((B.data * np.where(val == np.inf, 1, val), (B.row, B.col)), B.shape).tocsr())
    return Q, Bs

  def add_skeleton_projection(self, func, material_filter, grad = False):
    size = self.boundary_types.size
    proj_matrix = sparse.coo_array((size, size))
    proj_vector = np.zeros(size)
    for domain in (domain for domain in self.domains if domain.material in material_filter):
      for block in (block for material, _, elements in domain.boundaries for block in elements if material in material_filter[domain.material]):
        fem = domain.fabric(block, True)
        if grad:
          proj_matrix += fem.stiffness_matrix(proj_matrix.shape)
          proj_vector += fem.load_grad_vector(func, proj_vector.shape)
        else:
          proj_matrix += fem.mass_matrix(proj_matrix.shape)
          proj_vector += fem.load_vector(func, proj_vector.shape)

    if grad:
      proj_matrix = proj_matrix.tolil()
      proj_matrix[self.dirichlet_boundary, :] = 0
      proj_matrix[:, self.dirichlet_boundary] = 0
      proj_vector[self.dirichlet_boundary] = 0

    diag = proj_matrix.diagonal() + 1e-30
    proj, exit = sparse.linalg.cg(proj_matrix.tocsr(), proj_vector, M=sparse.diags(np.where(diag != 0, 1/diag, 1)), tol=1e-12, atol=0)
    assert exit == 0

    for domain, B in zip(self.domains, self.B):
      if domain.material in material_filter:
        self.g -= B[:, :domain.ext_dof_count] @ proj[domain.boundary_indices]

    return proj

  def assembly(self, dirichlet_name, materials):
    self.assembly_constraints(dirichlet_name)

    for domain in self.domains:
      domain.assembly(materials.get(domain.material, {}))

    def sparseMult(B, kernel):
      data = B.data * kernel[:, B.col]
      i = np.broadcast_to(B.row, data.shape)
      j = np.broadcast_to(np.arange(kernel.shape[0])[:, None], data.shape)
      return sparse.coo_array((data.flat, (i.flat, j.flat)), (B.shape[0], kernel.shape[0]))

    self.e = np.hstack([domain.kernel @ domain.load_vector for domain in self.domains])
    self.G = sparse.hstack([sparseMult(B.tocoo(), domain.kernel) for (domain, B) in zip(self.domains, self.B)], format='csr')

    self.g = np.zeros(self.dual_size)
    self.primal_size = self.e.size

  def decompose(self):
    for domain in self.domains:
      domain.decompose()

  def project(self, _lambda):
    return _lambda - self.Q @ (self.G @ (self.Coarse @ (self.G.T @ _lambda)))

  def projectT(self, _lambda):
    return _lambda - self.G @ (self.Coarse @ (self.G.T @ (self.Q @ _lambda)))

  def residual(self, _lambda):
    r = self.g.copy()
    for domain, B in zip(self.domains, self.B):
      r += B @ domain.solve_neumann(domain.load_vector - B.T @ _lambda)
    return r

  def operator(self, _lambda):
    r = np.zeros_like(_lambda)
    for domain, B in zip(self.domains, self.B):
      r += B @ domain.solve_neumann(B.T @ _lambda)
    return r

  def preconditioner(self, _lambda, lumped):
    r = np.zeros_like(_lambda)
    for domain, Bs in zip(self.domains, self.Bs):
      r += Bs @ domain.solve_dirichlet(Bs.T @ _lambda, lumped)
    return r

  def solutions(self, _lambda):
    solutions = []
    r = self.g.copy()
    for domain, B in zip(self.domains, self.B):
      x = domain.solve_neumann(domain.load_vector - B.T @ _lambda)
      r += B @ x
      solutions.append(x)
    alpha = -self.Coarse @ (self.G.T @ (self.Q @ r))
    alpha_split = np.hsplit(alpha, np.cumsum([domain.kernel.shape[0] for domain in self.domains[:-1]]))
    return [domain.calc_solution(x + a @ domain.kernel) for x, a, domain in zip(solutions, alpha_split, self.domains)]

  def prepare(self, precond, Q):
    if precond == "I":
      precond_func = lambda x: x
    elif precond == "Dirichlet":
      precond_func = lambda x: self.preconditioner(x, False)
    else:
      precond_func = lambda x: self.preconditioner(x, True)

    Qdiag, self.Bs = self.assembly_scaling()
    if Q == "M" and precond != "I":
      self.Q = sparse.linalg.LinearOperator(Qdiag.shape, matvec = precond_func, matmat = precond_func)
      self.Coarse = sparse.csr_array(np.linalg.inv(self.G.T @ (self.Q @ self.G.todense())))
    else:
      self.Q = Qdiag if Q == "Diag" else sparse.identity(self.dual_size)
      self.Coarse = sparse.csr_array(sparse.linalg.inv(self.G.T @ self.Q @ self.G))

    A = sparse.linalg.LinearOperator(Qdiag.shape, matvec = lambda x: self.projectT(self.operator(x)))
    M = sparse.linalg.LinearOperator(Qdiag.shape, matvec = lambda x: self.project(precond_func(x)))
    return A, M

  def solve(self, precond = "Dirichlet", Q = "Diag", tol=1e-7):
    i = 0
    def count_iter(x):
      nonlocal i
      i += 1

    A, M = self.prepare(precond, Q)
    lambda0 = self.Q @ (self.G @ (self.Coarse @ self.e))
    _lambda, exit = sparse.linalg.cg(A, self.projectT(self.residual(lambda0)), M=M, tol=tol, atol=0, callback=count_iter)
    assert exit == 0
    print(f'CG iters {i}')

    return self.solutions(lambda0 + _lambda)

class FetiProblemNotRed(FetiProblem):
  def build_constraints(self, links, constraints):
    idx = np.nonzero(links >= 0)[0]
    if idx[0] == 0:
      for j in idx[1:]:
        constraints[j - 1].append((self.dual_size, links[j], +1))
        self.dual_size += 1
      return

    for i in idx[:1]:
      for j in idx[1:]:
        if i < j:
          constraints[i - 1].append((self.dual_size, links[i], +1))
          constraints[j - 1].append((self.dual_size, links[j], -1))
          self.dual_size += 1

  def assembly_scaling(self):
    jumps = np.full((self.dual_size, 2), np.inf)
    q = np.full(self.dual_size, np.inf)
    scaling = np.zeros(self.boundary_types.shape)
    Bcoo = [B.tocoo() for B in self.B]
    for domain, B in zip(self.domains, Bcoo):
      scaling[domain.boundary_indices] += domain.scaling
      jumps[B.row, 1 * (B.data == -1)] = domain.scaling[B.col]
      H, h = domain.diameter
      q[B.row] = np.min((q[B.row], np.where(self.boundary_types[domain.boundary_indices[B.col]], h**(self.dim-2), (1 + np.log(H/h)*h**(self.dim-1)/H))), axis=0)

    Q = sparse.diags(q * np.min(jumps, axis=-1))
    Bd = []
    for domain, B in zip(self.domains, Bcoo):
      val = domain.scaling[B.col] / scaling[domain.boundary_indices[B.col]]
      Bd.append(sparse.coo_array((B.data / np.where(val == np.inf, 1, val), (B.row, B.col)), B.shape).tocsr())
    BBdt = sparse.linalg.inv(sparse.hstack(self.B, format='csc') @ sparse.hstack(Bd, format='csc').T)
    return Q, [BBdt @ b for b in Bd]