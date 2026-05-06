"""Finite-element domain assembly and local solves."""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from ._typing import Array, BoundaryBlock, ElementBlock, FloatArray, MaterialConfig
from .fem import FemLine2, FemTriangle3


class FemDomain:
    """Finite-element subdomain with boundary-first degrees of freedom."""

    def __init__(
        self,
        material: str | None,
        boundary_indices: ArrayLike,
        vertices: ArrayLike,
        elements: list[ElementBlock],
        boundaries: list[BoundaryBlock],
    ) -> None:
        """Create a finite-element domain from mesh blocks.

        Parameters
        ----------
        material
            Physical material name for the domain.
        boundary_indices
            Global skeleton indices for boundary vertices.
        vertices
            Domain vertices, ordered with boundary vertices first.
        elements
            Interior finite-element blocks.
        boundaries
            Boundary finite-element blocks with orientation tags.
        """
        self.material = material
        self.boundary_indices: Array = np.asarray(boundary_indices)
        self.vertices: Array = np.asarray(vertices)
        self.elements = elements
        self.boundaries = boundaries
        self.dof_count = self.vertices.shape[0]
        self.ext_dof_count = self.boundary_indices.size
        self.element_count = sum(len(e) for _, e, _, _ in self.elements)

    def fabric(self, data: ElementBlock, ext: bool = False) -> FemLine2 | FemTriangle3:
        """Create the finite-element helper for an element block."""
        indices = self.boundary_indices[data[1]] if ext else data[1]
        match data[0]:
            case "Line 2":
                return FemLine2(self.vertices[data[1]], indices, *data[2:])
            case "Triangle 3":
                return FemTriangle3(self.vertices[data[1]], indices, *data[2:])
            case _:
                raise Exception(f"Unsupported element type {data[0]}")

    def assembly(self, material_dict: MaterialConfig) -> None:
        """Assemble stiffness, load, scaling, and null-space data."""
        self.load_vector = np.zeros(self.dof_count)
        self.corr_vector = np.zeros_like(self.load_vector)
        self.kernel = np.ones((1, self.load_vector.size))  # only for Lagrange basis
        k = float(cast(float, material_dict.get("coeff", 1)))

        for bmat, tag, elements in self.boundaries:
            if f := material_dict.get(bmat):
                for fem in map(self.fabric, elements):
                    self.load_vector += np.sign(tag) * fem.load_vector(
                        f, self.load_vector.shape
                    )

        diameters = []
        self.scaling = np.full(self.ext_dof_count, k)
        self.stiffness_matrix = sparse.coo_array((self.dof_count, self.dof_count))
        for fem in map(self.fabric, self.elements):
            self.stiffness_matrix += k * fem.stiffness_matrix(
                self.stiffness_matrix.shape
            )
            self.corr_vector += fem.load_vector(1, self.corr_vector.shape)
            diameters.append(fem.diameter())
            if f := material_dict.get(self.material):
                self.load_vector += fem.load_vector(f, self.load_vector.shape)
        self.diameter = (
            (sumD := sum(D for D, d in diameters)),
            sum(d * D for D, d in diameters) / sumD,
        )

        self.stiffness_matrix = self.stiffness_matrix.tocsc()  # tocsr()
        self.neumann_factor = None
        self.dirichlet_factor = None

    def decompose(self) -> None:
        """Factor matrices used by Neumann and Dirichlet local solves."""
        a = self.stiffness_matrix[0, 0]
        self.stiffness_matrix[0, 0] *= 2
        self.neumann_factor = sparse.linalg.factorized(self.stiffness_matrix)
        self.stiffness_matrix[0, 0] = a
        self.dirichlet_factor = sparse.linalg.factorized(
            self.stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :]
        )

    def solve_neumann(self, flow: Array) -> FloatArray:
        """Solve the local Neumann problem for the supplied load vector."""

        def mult(x):
            return self.stiffness_matrix @ x + self.corr_vector * (self.corr_vector @ x)

        if self.neumann_factor:
            return self.neumann_factor(flow)
        return sparse.linalg.minres(
            sparse.linalg.LinearOperator(
                self.stiffness_matrix.shape,
                matvec=mult,
                dtype=self.stiffness_matrix.dtype,
            ),
            flow,
            rtol=1e-12,
        )[0]

    def solve_dirichlet(self, disp: Array, lumped: bool = False) -> FloatArray:
        """Recover local interface flow from prescribed boundary values."""
        flow = (
            self.stiffness_matrix[:, : self.ext_dof_count] @ disp[: self.ext_dof_count]
        )
        if not lumped:
            if self.dirichlet_factor:
                sol = self.dirichlet_factor(flow[self.ext_dof_count :])
            else:
                sol = sparse.linalg.minres(
                    self.stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :],
                    flow[self.ext_dof_count :],
                    rtol=1e-12,
                )[0]
            flow -= self.stiffness_matrix[:, self.ext_dof_count :] @ sol
        return flow

    def calc_solution(self, sol: Array) -> Array:
        """Return the finite-element solution values for all domain vertices."""
        return sol
