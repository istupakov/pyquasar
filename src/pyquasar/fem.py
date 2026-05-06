"""Finite-element integration helpers for line and triangle elements."""

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from ._typing import Array, FieldFunction, FloatArray, Shape


class FemBase:
    """Base class for local finite-element vector and matrix assembly."""

    def __init__(self, elements: ArrayLike, quad: ArrayLike, weight: ArrayLike) -> None:
        self.elements: Array = np.asarray(elements)
        self.quad: Array = np.asarray(quad)
        self.weight: Array = np.asarray(weight)

    def perp(self, vec: Array) -> FloatArray:
        """Return vectors rotated by 90 degrees in the element plane."""
        res = vec[..., ::-1].copy()
        np.negative(res[..., 0], out=res[..., 0])
        return res

    def vector(self, data: Array, shape: Shape) -> FloatArray:
        """Assemble local element data into a global vector."""
        res = np.zeros(shape)
        np.add.at(res, self.elements, data)
        return res

    def matrix(self, data: Array, shape: Shape) -> sparse.coo_array:
        """Assemble local element data into a global sparse matrix."""
        i = np.broadcast_to(self.elements[:, None, :], data.shape)
        j = np.broadcast_to(self.elements[:, :, None], data.shape)
        return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)


class FemBase1D(FemBase):
    """Base class for line-element geometry and quadrature data."""

    def __init__(
        self,
        elem_vert: ArrayLike,
        elements: ArrayLike,
        quad: ArrayLike,
        weight: ArrayLike,
    ) -> None:
        elem_vert = np.asarray(elem_vert)
        super().__init__(
            elements, 0.5 * (1 + np.asarray(quad)), 0.5 * np.asarray(weight)
        )
        self.center = elem_vert[:, 0]
        self.dir = elem_vert[:, 1] - self.center
        self.J = np.linalg.norm(self.dir, axis=-1)[:, None]
        self.normal = -self.perp(self.dir) / self.J

    def diameter(self) -> tuple[float, float]:
        """Return total and mean line-element diameters."""
        return np.sum(self.J), np.mean(self.J)


class FemBase2D(FemBase):
    """Base class for triangle-element geometry and quadrature data."""

    def __init__(
        self,
        elem_vert: ArrayLike,
        elements: ArrayLike,
        quad: ArrayLike,
        weight: ArrayLike,
    ) -> None:
        elem_vert = np.asarray(elem_vert)
        super().__init__(elements, quad, weight)
        self.center = elem_vert[:, 0]
        self.dir1 = elem_vert[:, 1] - self.center
        self.dir2 = elem_vert[:, 2] - self.center
        self.normal = (
            self.dir1[:, 0] * self.dir2[:, 1] - self.dir1[:, 1] * self.dir2[:, 0]
        ).reshape(self.dir1.shape[0], -1)
        self.J = np.linalg.norm(self.normal, axis=-1)[:, None]
        self.normal /= self.J
        codir = [-self.perp(self.dir2), self.perp(self.dir1)] / self.J
        self.cometric = np.sum(codir[:, None, :] * codir[None, :, :], axis=-1)

    def diameter(self) -> tuple[float, float]:
        """Return characteristic total and mean triangle diameters."""
        return np.sum(self.J) ** 0.5, np.mean(self.J) ** 0.5


class FemLine2(FemBase1D):
    """Two-node line element with linear Lagrange basis functions."""

    def mass_matrix(self, shape: Shape) -> sparse.coo_array:
        """Assemble the line-element mass matrix."""
        psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
        return self.matrix(
            self.J[..., None] * ((psi[None, :] * psi[:, None]) @ self.weight), shape
        )

    def stiffness_matrix(self, shape: Shape) -> sparse.coo_array:
        """Assemble the line-element stiffness matrix."""
        psiGrad = np.array([-1, 1])
        return self.matrix(
            (psiGrad[None, :] * psiGrad[:, None]) / self.J[..., None], shape
        )

    def skew_grad_matrix(self, shape: Shape) -> sparse.coo_array:
        """Assemble the line-element skew-gradient matrix."""
        psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
        psiGrad = np.array([-1, 1])[..., None]
        return self.matrix(
            np.ones_like(self.J[..., None])
            * ((psi[None, :] * psiGrad[:, None]) @ self.weight),
            shape,
        )

    def load_vector(
        self,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike] | ArrayLike,
        shape: Shape,
    ) -> FloatArray:
        """Assemble a line load vector from a scalar boundary field."""
        psi = np.array([1 - self.quad[:, 0], self.quad[:, 0]])
        if callable(func):
            f = cast(FieldFunction, func)(
                self.center[:, None] + self.quad[None, :, 0, None] * self.dir[:, None],
                self.normal[:, None],
            )
        else:
            f = func
        return self.vector(
            self.J * ((psi * np.atleast_1d(f)[:, None]) @ self.weight), shape
        )

    def load_grad_vector(
        self,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike] | ArrayLike,
        shape: Shape,
    ) -> FloatArray:
        """Assemble a line load vector from a vector gradient field."""
        psiGrad = np.array([-1, 1])
        if callable(func):
            f = cast(FieldFunction, func)(
                self.center[:, None] + self.quad[None, :, 0, None] * self.dir[:, None],
                self.normal[:, None],
            )
        else:
            f = func
        return self.vector(
            (
                psiGrad
                * np.sum(
                    np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weight,
                    axis=-1,
                )[:, None]
            )
            / self.J,
            shape,
        )


class FemTriangle3(FemBase2D):
    """Three-node triangle element with linear Lagrange basis functions."""

    def mass_matrix(self, shape: Shape) -> sparse.coo_array:
        """Assemble the triangle-element mass matrix."""
        psi = np.array(
            [1 - self.quad[:, 0] - self.quad[:, 1], self.quad[:, 0], self.quad[:, 1]]
        )
        return self.matrix(
            self.J[..., None] * ((psi[None, :] * psi[:, None]) @ self.weight), shape
        )

    def stiffness_matrix(self, shape: Shape) -> sparse.coo_array:
        """Assemble the triangle-element stiffness matrix."""
        psiGrad = np.array([[-1, 1, 0], [-1, 0, 1]])
        S = 0.5 * psiGrad[:, None, :, None] * psiGrad[None, :, None, :]
        return self.matrix(
            self.J[..., None]
            * np.sum(self.cometric[:, :, None, None] * S[..., None], axis=(0, 1)).T,
            shape,
        )

    def load_vector(
        self,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike] | ArrayLike,
        shape: Shape,
    ) -> FloatArray:
        """Assemble a triangle load vector from a scalar source field."""
        psi = np.array(
            [1 - self.quad[:, 0] - self.quad[:, 1], self.quad[:, 0], self.quad[:, 1]]
        )
        if callable(func):
            point = (
                self.center[:, None]
                + self.quad[None, :, 0, None] * self.dir1[:, None]
                + self.quad[None, :, 1, None] * self.dir2[:, None]
            )
            f = cast(FieldFunction, func)(point, self.normal[:, None])
        else:
            f = func
        return self.vector(
            self.J * ((psi * np.atleast_1d(f)[:, None]) @ self.weight), shape
        )
