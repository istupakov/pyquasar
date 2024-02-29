from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .fem_domain import FemDomain
from .bem_domain import BemDomain


class FetiProblem:
  """FETI problem class."""

  def __init__(self, domains: List[FemDomain | BemDomain], dim: int = 2):
    self._domains = domains

  @property
  def dim(self) -> int:
    """Dimension of the FETI problem."""
    return self.domains[0].vertices.shape[-1]

  @property
  def domains(self) -> List[FemDomain | BemDomain]:
    """List of domains of the FETI problem."""
    return self._domains

  @property
  def dual_size(self) -> int:
    """Size of the dual space."""
    return self._dual_size

  @property
  def boundary_types(self) -> npt.NDArray[np.bool_]:
    """Array of boundary types."""
    return self._boundary_types

  @property
  def dirichlet_boundary(self) -> npt.NDArray[np.bool_]:
    """Array of dirichlet boundary nodes."""
    return self._dirichlet_boundary

  @property
  def B(self) -> list[sparse.csr_array]:
    """List of communication condition matrices."""
    if hasattr(self, "_B"):
      return self._B
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def e(self) -> npt.NDArray[np.floating]:
    """Global load vector of primal space."""
    if hasattr(self, "_e"):
      return self._e
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def G(self) -> sparse.csr_array:
    """Dual space matrix."""
    if hasattr(self, "_G"):
      return self._G
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def g(self) -> npt.NDArray[np.floating]:
    """Dual space vector."""
    if hasattr(self, "_g"):
      return self._g
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def primal_size(self) -> int:
    """Size of the primal space."""
    if hasattr(self, "_primal_size"):
      return self._primal_size
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def Q(self) -> sparse.dia_matrix:
    """Preconditioner matrix for orthogonal projection operator."""
    if hasattr(self, "_Q"):
      return self._Q
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def Bs(self) -> list[sparse.csr_array]:
    """List of scaled communication condition matrices."""
    if hasattr(self, "_Bs"):
      return self._Bs
    else:
      raise AttributeError("Problem is not assembled yet.")

  @property
  def coarse(self) -> sparse.csr_array:
    """Coarse space matrix."""
    if hasattr(self, "_coarse"):
      return self._coarse
    else:
      raise AttributeError("Problem is not assembled yet.")

  def _build_constraints(self, links: npt.NDArray[np.signedinteger], constraints: list[list[tuple[int, int, int]]]) -> None:
    """Build constraints for the FETI problem.

    Parameters
    ----------
    links: NDArray[int]
      Array of links.
    constraints: list[list[tuple[int, int, int]]]
      List of constraints.
    """
    indices = np.nonzero(links >= 0)[0]
    if indices[0] == 0:
      for j in indices[1:]:
        constraints[j - 1].append((self.dual_size, links[j], +1))
        self._dual_size += 1
      return

    for i in indices:
      for j in indices:
        if i < j:
          constraints[i - 1].append((self.dual_size, links[i], +1))
          constraints[j - 1].append((self.dual_size, links[j], -1))
          self._dual_size += 1

  def _assembly_constraints(self, dirichlet_name: str) -> None:
    """Assemble constraints for the FETI problem.

    Parameters
    ----------
    dirichlet_name: str
      Name of the dirichlet boundary.
    """
    max_index = max(domain.boundary_indices.max() for domain in self.domains)
    connections = np.full((int(max_index + 1), len(self.domains) + 1), -1, dtype=np.int_)

    for i, domain in enumerate(self.domains):
      dir_internal_indices = [
        block.node_tags for boundary in domain.boundaries for block in boundary.elements if boundary.type == dirichlet_name
      ]
      dirichlet = domain.boundary_indices[np.unique(np.concatenate(dir_internal_indices))] if dir_internal_indices else []
      connections[dirichlet, 0] = dirichlet
      boundary = domain.boundary_indices
      connections[boundary, i + 1] = np.arange(boundary.size)

    self._boundary_types = np.sum(connections >= 0, axis=-1) > 2
    self._dirichlet_boundary = connections[:, 0] >= 0

    constraints: list[list] = [[] for _ in range(connections.shape[1] - 1)]
    self._dual_size = 0
    for links in connections:
      self._build_constraints(links, constraints)

    self._B = []
    for constraint, domain in zip(constraints, self.domains):
      i, j, v = np.array(constraint).T
      self._B.append(sparse.coo_array((v, (i, j)), (self.dual_size, domain.dof_count)).tocsr())

  def condition_number_estimate(self) -> float:
    """Estimate the condition number of the FETI problem.

    Returns
    -------
    float
    """
    return np.log1p(max(d.diameter[0] / d.diameter[1] for d in self.domains)) ** 2

  def _assembly_scaling(self) -> tuple[sparse.dia_matrix, list[sparse.csr_array]]:
    """Assemble scaling for the FETI problem.

    Returns
    -------
    tuple[sparse.dia_matrix, list[sparse.csr_array]]
      Tuple of precondition matrix Q and list of scaled communication condition matrices Bs.
    """
    jumps = np.full((self.dual_size, 2), np.inf)
    q = np.full(self.dual_size, np.inf)
    scaling = np.zeros(self.boundary_types.shape)
    Bcoo = [B.tocoo() for B in self.B]
    for domain, B in zip(self.domains, Bcoo):
      scaling[domain.boundary_indices] += domain.scaling
      jumps[B.row, 1 * (B.data == -1)] = domain.scaling[B.col]
      H, h = domain.diameter
      q[B.row] = np.min(
        (
          q[B.row],
          np.where(
            self.boundary_types[domain.boundary_indices[B.col]],
            h ** (self.dim - 2),
            (1 + np.log(H / h) * h ** (self.dim - 1) / H),
          ),
        ),
        axis=0,
      )

    Q = sparse.diags(q * np.min(jumps, axis=-1))
    Bs = []
    for domain, B in zip(self.domains, Bcoo):
      val = jumps[B.row, 1 * (B.data == 1)] / scaling[domain.boundary_indices[B.col]]
      Bs.append(sparse.coo_array((B.data * np.where(val == np.inf, 1, val), (B.row, B.col)), B.shape).tocsr())
    return Q, Bs

  def add_skeleton_projection(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    material_filter: dict[str, set[str]],
    grad: bool = False,
  ) -> npt.NDArray[np.floating]:
    """Add skeleton projection to the FETI problem.

    Parameters
    ----------
    func: Callable[[NDArray[float], NDArray[float]], NDArray[float]] | ArrayLike
      Function that is projected onto the basis of the problem.
    material_filter: dict[str, set[str]]
      Dictionary of materials and boundary types.
    grad: bool
      Gradient flag.

    Returns
    -------
    NDArray[float]
    """
    size = self.boundary_types.size
    proj_matrix = sparse.coo_array((size, size))
    proj_vector = np.zeros(size)
    for domain in (domain for domain in self.domains if domain.material in material_filter):
      for block in (
        block for boundary in domain.boundaries for block in boundary.elements if boundary.type in material_filter[domain.material]
      ):
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
    proj, exit = sparse.linalg.cg(proj_matrix.tocsr(), proj_vector, M=sparse.diags(np.where(diag != 0, 1 / diag, 1)), tol=1e-12, atol=0)
    assert exit == 0, "CG did not converge."

    for domain, B in zip(self.domains, self.B):
      if domain.material in material_filter:
        self._g -= B[:, : domain.ext_dof_count] @ proj[domain.boundary_indices]

    return proj

  def assembly(
    self,
    dirichlet_name: str,
    materials: dict[
      str,
      dict[Optional[str], Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike],
    ],
  ) -> None:
    """Assemble the FETI problem.

    Parameters
    ----------
    dirichlet_name: str
      Name of the dirichlet boundary.
    materials: dict[str, dict[Optional[str], Callable[[NDArray[float], NDArray[float]], NDArray[float]] | ArrayLike]
      Materials dictionary with material name, boundary conditions and source function.
    """
    self._assembly_constraints(dirichlet_name)

    for domain in self.domains:
      domain.assembly(materials.get(domain.material, {}))

    def sparse_mult(B, kernel: npt.NDArray[np.floating]) -> sparse.coo_array:
      data = B.data * kernel[:, B.col]
      i = np.broadcast_to(B.row, data.shape)
      j = np.broadcast_to(np.arange(kernel.shape[0])[:, None], data.shape)
      return sparse.coo_array((data.flat, (i.flat, j.flat)), (B.shape[0], kernel.shape[0]))

    self._e = np.hstack([domain.kernel @ domain.load_vector for domain in self.domains])
    self._G = sparse.hstack([sparse_mult(B.tocoo(), domain.kernel) for (domain, B) in zip(self.domains, self.B)], format="csr")

    self._g = np.zeros(self.dual_size)
    self._primal_size = self.e.size

  def decompose(self) -> None:
    """Decompose all domains in the problem."""
    for domain in self.domains:
      domain.decompose()

  def project(self, lambda_: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Project the lagrange multipliers to the coarse space.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.

    Returns
    -------
    NDArray[float]
    """
    return lambda_ - self.Q @ (self.G @ (self.coarse @ (self.G.T @ lambda_)))

  def project_transposed(self, lambda_: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Project the transposed lagrange multipliers to the coarse space.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.

    Returns
    -------
    NDArray[float]
    """
    return lambda_ - self.G @ (self.coarse @ (self.G.T @ (self.Q @ lambda_)))

  def residual(self, lambda_: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the residual.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.
    Returns
    -------
    NDArray[float]
    """
    r = self.g.copy()
    for domain, B in zip(self.domains, self.B):
      r += B @ domain.solve_neumann(domain.load_vector - B.T @ lambda_)
    return r

  def operator(self, lambda_: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the problem operator.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.

    Returns
    -------
    NDArray[float]
    """
    r = np.zeros_like(lambda_)
    for domain, B in zip(self.domains, self.B):
      r += B @ domain.solve_neumann(B.T @ lambda_)
    return r

  def preconditioner(self, lambda_: npt.NDArray[np.floating], lumped: bool) -> npt.NDArray[np.floating]:
    """Compute the preconditioner.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.
    lumped: bool
      Lumped preconditioner flag.

    Returns
    -------
    NDArray[float]
    """
    r = np.zeros_like(lambda_)
    for domain, Bs in zip(self.domains, self.Bs):
      r += Bs @ domain.solve_dirichlet(Bs.T @ lambda_, lumped)
    return r

  def solutions(self, lambda_: npt.NDArray[np.floating]) -> list[npt.NDArray[np.floating]]:
    """Compute the solutions.

    Parameters
    ----------
    lambda_: NDArray[float]
      Lagrange multipliers vector.

    Returns
    -------
    list[NDArray[float]]
    """
    solutions = []
    r = self.g.copy()
    for domain, B in zip(self.domains, self.B):
      x = domain.solve_neumann(domain.load_vector - B.T @ lambda_)
      r += B @ x
      solutions.append(x)
    alpha = -self.coarse @ (self.G.T @ (self.Q @ r))
    alpha_split = np.hsplit(alpha, np.cumsum([domain.kernel.shape[0] for domain in self.domains[:-1]]))
    return [domain.calc_solution(x + a @ domain.kernel) for x, a, domain in zip(solutions, alpha_split, self.domains)]

  def prepare(self, precond: str, Q: str) -> tuple[sparse.linalg.LinearOperator, sparse.linalg.LinearOperator]:
    """Prepare the preconditioners for FETI problem solving

    Parameters
    ----------
    precond: str
      Type preconditioner matrix M for solver.
    Q: str
      Type of preconditioner matrix Q for orhogonal projection operator.

    Returns
    -------
    tuple[sparse.linalg.LinearOperator, sparse.linalg.LinearOperator]
      Problem operator and preconditioner.
    """
    if precond == "I":
      precond_func = lambda x: x  # noqa: E731
    elif precond == "Dirichlet":
      precond_func = lambda x: self.preconditioner(x, False)  # noqa: E731
    else:
      precond_func = lambda x: self.preconditioner(x, True)  # noqa: E731

    Qdiag, self._Bs = self._assembly_scaling()
    if Q == "M" and precond != "I":
      self._Q = sparse.linalg.LinearOperator(Qdiag.shape, matvec=precond_func, matmat=precond_func)
      self._coarse = sparse.csr_array(np.linalg.inv(self.G.T @ (self.Q @ self.G.todense())))
    else:
      self._Q = Qdiag if Q == "Diag" else sparse.identity(self.dual_size)
      self._coarse = sparse.csr_array(sparse.linalg.inv(self.G.T @ self.Q @ self.G))

    A = sparse.linalg.LinearOperator(Qdiag.shape, matvec=lambda x: self.project_transposed(self.operator(x)))
    M = sparse.linalg.LinearOperator(Qdiag.shape, matvec=lambda x: self.project(precond_func(x)))
    return A, M

  def solve(self, precond: str = "Dirichlet", Q: str = "Diag", rtol: float = 1e-7, atol: float = 0.0) -> list[npt.NDArray[np.floating]]:
    """Solve the FETI problem.

    Parameters
    ----------
    precond: str, optional
      Type preconditioner matrix M for solver, by default "Dirichlet".
    Q: str, optional
      Type of preconditioner matrix Q for orhogonal projection operator, by default "Diag".
    rtol, atol: float, optional
      Parameters for the convergence test. For convergence, `norm(b - A @ x) <= max(rtol*norm(b), atol)` should be satisfied. The default is `atol=0`. and `rtol=1e-7`.

    Returns
    -------
    list[NDArray[float]]

    Raises
    ------
    AssertionError:
      If solver fails to converge.
    """
    i = 0

    def count_iter(x: npt.NDArray[np.float_]) -> None:
      nonlocal i
      i += 1

    A, M = self.prepare(precond, Q)
    lambda0 = self.Q @ (self.G @ (self.coarse @ self.e))
    lambda_, exit = sparse.linalg.cg(A, self.project_transposed(self.residual(lambda0)), M=M, rtol=rtol, atol=atol, callback=count_iter)
    assert exit == 0
    print(f"CG iters {i}")

    return self.solutions(lambda0 + lambda_)


class FetiProblemNotRed(FetiProblem):
  def _build_constraints(self, links: npt.NDArray[np.signedinteger], constraints: list[list[tuple[int, int, int]]]) -> None:
    indices = np.nonzero(links >= 0)[0]
    if indices[0] == 0:
      for j in indices[1:]:
        constraints[j - 1].append((self.dual_size, links[j], +1))
        self._dual_size += 1
      return

    for i in indices[:1]:
      for j in indices[1:]:
        if i < j:
          constraints[i - 1].append((self.dual_size, links[i], +1))
          constraints[j - 1].append((self.dual_size, links[j], -1))
          self._dual_size += 1

  def _assembly_scaling(self) -> tuple[sparse.dia_matrix, list[sparse.csr_array]]:
    jumps = np.full((self.dual_size, 2), np.inf)
    q = np.full(self.dual_size, np.inf)
    scaling = np.zeros(self.boundary_types.shape)
    Bcoo = [B.tocoo() for B in self.B]
    for domain, B in zip(self.domains, Bcoo):
      scaling[domain.boundary_indices] += domain.scaling
      jumps[B.row, 1 * (B.data == -1)] = domain.scaling[B.col]
      H, h = domain.diameter
      q[B.row] = np.min(
        (
          q[B.row],
          np.where(
            self.boundary_types[domain.boundary_indices[B.col]],
            h ** (self.dim - 2),
            (1 + np.log(H / h) * h ** (self.dim - 1) / H),
          ),
        ),
        axis=0,
      )

    Q = sparse.diags(q * np.min(jumps, axis=-1))
    Bd = []
    for domain, B in zip(self.domains, Bcoo):
      val = domain.scaling[B.col] / scaling[domain.boundary_indices[B.col]]
      Bd.append(sparse.coo_array((B.data / np.where(val == np.inf, 1, val), (B.row, B.col)), B.shape).tocsr())
    BBdt = sparse.linalg.inv(sparse.hstack(self.B, format="csc") @ sparse.hstack(Bd, format="csc").T)
    return Q, [BBdt @ b for b in Bd]
