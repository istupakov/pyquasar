"""Shared type aliases for pyquasar's numerical APIs."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Array = NDArray[Any]
type FloatArray = NDArray[np.floating]
type IntArray = NDArray[np.integer]
type BoolArray = NDArray[np.bool_]
type Scalar = float | int | np.number
type Shape = int | tuple[int, ...]

type FieldFunction = Callable[[ArrayLike, ArrayLike], ArrayLike]
type MaterialValue = Scalar | FieldFunction
type MaterialConfig = Mapping[str | None, MaterialValue]
type Materials = Mapping[str | None, MaterialValue | MaterialConfig]

type ElementBlock = tuple[str, IntArray, FloatArray, FloatArray]
type BoundaryBlock = tuple[str | None, int, list[ElementBlock]]
type DomainData = tuple[
    str | None,
    IntArray,
    FloatArray,
    list[ElementBlock],
    list[BoundaryBlock],
]

type ElementData = Sequence[object]
type ProjectionFilter = Mapping[str | None, Iterable[str | None]]
