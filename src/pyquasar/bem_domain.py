"""Boundary-element domain assembly and local solves."""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._typing import Array, BoundaryBlock, ElementBlock, FloatArray, MaterialConfig
from .bem import BemLine2


class BemDomain:
    """Boundary-element subdomain defined by the boundary of a mesh region."""

    def __init__(
        self,
        material: str | None,
        boundary_indices: ArrayLike,
        vertices: ArrayLike,
        elements: list[ElementBlock],
        boundaries: list[BoundaryBlock],
    ) -> None:
        """Create a boundary-element domain from mesh blocks.

        Parameters
        ----------
        material
            Physical material name for the domain.
        boundary_indices
            Global skeleton indices for boundary vertices.
        vertices
            Domain vertices, ordered with boundary vertices first.
        elements
            Unused interior element blocks retained for API compatibility.
        boundaries
            Oriented boundary element blocks used to build the BEM mesh.
        """
        self.material = material
        self.boundary_indices: Array = np.asarray(boundary_indices)
        self.vertices: Array = np.asarray(vertices)
        self.elements = elements
        self.boundaries = boundaries
        self.dof_count = self.boundary_indices.size
        self.ext_dof_count = self.boundary_indices.size

        bem_elements = [
            block[1] if tag > 0 else block[1][..., ::-1]
            for bmat, tag, elements in self.boundaries
            for block in elements
        ]
        self.bem_elements = (
            self.boundaries[0][2][0][0],
            np.concatenate(bem_elements),
            self.boundaries[0][2][0][2],
            self.boundaries[0][2][0][3],
        )
        self.element_count = len(self.bem_elements[1])

    def fabric(self, data: ElementBlock, ext: bool = False) -> BemLine2:
        """Create the boundary-element helper for an element block."""
        indices = self.boundary_indices[data[1]] if ext else data[1]
        match data[0]:
            case "Line 2":
                return BemLine2(self.vertices[data[1]], indices, *data[2:])
            case _:
                raise Exception(f"Unsupported element type {data[0]}")

    def assembly(self, material_dict: MaterialConfig) -> None:
        """Assemble BEM operators, load vectors, and post-processing weights."""
        self.load_vector = np.zeros(self.dof_count)
        self.kernel = np.ones((1, self.load_vector.size))
        k = float(cast(float, material_dict.get("coeff", 1)))
        self.scaling = np.full(self.ext_dof_count, k)

        for bmat, tag, elements in self.boundaries:
            if f := material_dict.get(bmat):
                for fem in map(self.fabric, elements):
                    self.load_vector += np.sign(tag) * fem.load_vector(
                        f, self.load_vector.shape
                    )

        bem = self.fabric(self.bem_elements)
        self.diameter = bem.diameter()
        V, K, self.D = bem.bem_matrices(5)
        M = bem.mass_matrix(K.shape, 0, 1)
        K += M / 2

        invV = np.linalg.inv(V)
        self.invVK = invV @ K

        corr0 = bem.load_vector(1, V.shape[0], 0)
        w = invV @ corr0
        alpha = k / (4 * (w @ corr0))
        corr_vector = M @ w

        self.S = k * (self.D + K.T @ self.invVK)
        self.D *= k
        self.invS = np.linalg.inv(self.S + alpha * np.outer(corr_vector, corr_vector))

        if source := material_dict.get(self.material):
            assert np.isscalar(source) and np.isreal(source)
            f = float(cast(float, source))
            N0 = f * bem.load_vector(lambda p, n: bem.newton(p, 0), V.shape[0], 0)

            def normal_newton_trace(p, n):
                trace = bem.newton(p, 1)
                normals = np.asarray(n)
                return trace[..., 0] * normals[..., 0] + trace[..., 1] * normals[..., 1]

            N1 = f * bem.load_vector(
                normal_newton_trace,
                self.load_vector.shape,
                1,
            )
            self.invVN = invV @ N0
            self.load_vector += K.T @ self.invVN - N1
            self.f = f
        else:
            self.invVN = 0
            self.f = 0

        self.vertices_weights = bem.result_weights(self.vertices)

    def decompose(self) -> None:
        """Keep the domain API compatible with finite-element domains."""
        pass

    def solve_neumann(self, flow: Array) -> FloatArray:
        """Solve the local Neumann problem using the inverse Steklov operator."""
        return self.invS @ flow

    def solve_dirichlet(self, disp: Array, lumped: bool = False) -> FloatArray:
        """Recover boundary flow from prescribed boundary potential values."""
        if lumped:
            return self.D @ disp
        else:
            return self.S @ disp

    def calc_solution(self, q: Array) -> FloatArray:
        """Evaluate the reconstructed potential at all stored vertices."""
        p = self.invVK @ q - self.invVN
        res = (
            self.vertices_weights[0] @ p
            - self.vertices_weights[1] @ q
            + self.vertices_weights[2] * self.f
        )
        res[: q.size] = q
        return res
