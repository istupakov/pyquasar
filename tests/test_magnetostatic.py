from pathlib import Path

import numpy as np
import pytest

from pyquasar import FemDomain, FetiProblem, load_mesh

CIRCULAR_CURRENT_GEO = (
    Path(__file__).resolve().parents[1] / "examples" / "circular_current.geo"
)

MU0 = 4 * np.pi * 1e-7
RADIUS = 1.0
OUTER_RADIUS = 4.0
TOTAL_CURRENT = 1.0
CURRENT_DENSITY = TOTAL_CURRENT / (np.pi * RADIUS**2)
PRECONDITIONER_Q_CASES = [
    ("I", "I"),
    ("Dirichlet", "I"),
    ("Lumped Dirichlet", "I"),
    ("Dirichlet", "Diag"),
    ("Lumped Dirichlet", "Diag"),
]


def radius(points):
    return np.linalg.norm(points, axis=-1)


def analytic_A(points, normals=0):
    r = radius(points)
    inside = r <= RADIUS
    values = np.empty_like(r)
    values[inside] = (
        MU0
        * CURRENT_DENSITY
        * (RADIUS**2 * (0.5 + np.log(OUTER_RADIUS / RADIUS)) - 0.5 * r[inside] ** 2)
        / 2
    )
    values[~inside] = (
        MU0 * TOTAL_CURRENT * np.log(OUTER_RADIUS / r[~inside]) / (2 * np.pi)
    )
    return values


def analytic_grad_A(points):
    points = np.asarray(points)
    r = radius(points)
    scale = np.empty_like(r)
    inside = r <= RADIUS
    scale[inside] = -MU0 * CURRENT_DENSITY / 2
    scale[~inside] = -MU0 * TOTAL_CURRENT / (2 * np.pi * r[~inside] ** 2)
    return scale[..., None] * points


def analytic_B(points):
    grad = analytic_grad_A(points)
    return np.stack((grad[..., 1], -grad[..., 0]), axis=-1)


def analytic_H(points, normals=0):
    return analytic_B(points) / MU0


def normal_derivative_A(points, normals):
    return np.sum(analytic_grad_A(points) * normals, axis=-1)


def magnetic_flux(points, normals):
    return np.sum(analytic_B(points) * normals, axis=-1)


def zero(points, normals=0):
    return np.zeros(np.shape(points)[:-1])


def load_domains(materials=None, num_part=4, refine=0):
    domains = [
        FemDomain(*data)
        for data in load_mesh(str(CIRCULAR_CURRENT_GEO), num_part, refine)
    ]
    if materials is not None:
        domains = [domain for domain in domains if domain.material in materials]

    used_indices = np.unique(
        np.concatenate([domain.boundary_indices for domain in domains])
    )
    remap = np.empty(used_indices.max() + 1, dtype=int)
    remap[used_indices] = np.arange(used_indices.size)
    for domain in domains:
        domain.boundary_indices = remap[domain.boundary_indices]
    return domains


def build_problem(domains, dirichlet_name, materials, projections=()):
    problem = FetiProblem(domains)
    problem.assembly(dirichlet_name, materials)
    for func, material_filter, grad in projections:
        problem.add_skeleton_projection(func, material_filter, grad=grad)
    problem.decompose()
    return problem


def relative_solution_error(problem, solutions, expected_func):
    actual_norm = 0.0
    error_norm = 0.0
    for domain, solution in zip(problem.domains, solutions):
        expected = expected_func(domain.vertices)
        actual_norm += np.linalg.norm(expected) ** 2
        error_norm += np.linalg.norm(solution - expected) ** 2
    return (error_norm / actual_norm) ** 0.5


def absolute_solution_norm(problem, solutions):
    norm = 0.0
    count = 0
    for domain, solution in zip(problem.domains, solutions):
        norm += np.linalg.norm(solution) ** 2
        count += solution.size
    return (norm / count) ** 0.5


def build_total_A_problem(refine=0):
    materials = {
        "coil": {"coeff": 1.0 / MU0, "coil": CURRENT_DENSITY},
        "air": {"coeff": 1.0 / MU0},
        "dirichlet": analytic_A,
    }
    return build_problem(
        load_domains(refine=refine),
        "dirichlet",
        materials,
        [(analytic_A, {"coil": {"dirichlet"}, "air": {"dirichlet"}}, False)],
    )


@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_total_A_circular_current(preconditioner, q_mode):
    problem = build_total_A_problem()

    solutions = problem.solve(preconditioner, q_mode)

    assert relative_solution_error(problem, solutions, analytic_A) <= 3e-2


def test_total_A_refinement_convergence():
    errors = []
    for refine in range(3):
        problem = build_total_A_problem(refine)
        solutions = problem.solve("Dirichlet", "Diag")
        errors.append(relative_solution_error(problem, solutions, analytic_A))

    for coarse_error, refined_error in zip(errors, errors[1:]):
        assert refined_error < coarse_error
        assert coarse_error / refined_error >= 2.0


@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_reduced_A_circular_current(preconditioner, q_mode):
    materials = {
        "air": {
            "coeff": 1.0 / MU0,
            "gap": lambda p, n: normal_derivative_A(p, n) / MU0,
        },
        "dirichlet": zero,
    }
    problem = build_problem(
        load_domains({"air"}),
        "dirichlet",
        materials,
        [(zero, {"air": {"dirichlet"}}, False)],
    )

    solutions = problem.solve(preconditioner, q_mode)

    assert relative_solution_error(problem, solutions, analytic_A) <= 3e-2


@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_mixed_A_circular_current(preconditioner, q_mode):
    materials = {
        "coil": {"coeff": 1.0 / MU0, "coil": CURRENT_DENSITY},
        "air": {
            "coeff": 1.0 / MU0,
            "gap": lambda p, n: normal_derivative_A(p, n) / MU0,
        },
        "dirichlet": zero,
    }
    problem = build_problem(
        load_domains(),
        "dirichlet",
        materials,
        [
            (zero, {"coil": {"dirichlet"}, "air": {"dirichlet"}}, False),
            (analytic_A, {"air": {"gap"}}, False),
        ],
    )

    solutions = problem.solve(preconditioner, q_mode)

    for domain, solution in zip(problem.domains, solutions):
        if domain.material == "coil":
            expected = analytic_A(domain.vertices)
            assert (
                np.linalg.norm(solution - expected) / np.linalg.norm(expected) <= 5e-2
            )
        else:
            expected = 2 * analytic_A(domain.vertices)
            assert (
                np.linalg.norm(solution - expected) / np.linalg.norm(expected) <= 3e-2
            )


@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_reduced_phi_circular_current(preconditioner, q_mode):
    materials = {
        "air": {
            "coeff": MU0,
            "gap": magnetic_flux,
        },
        "dirichlet": zero,
    }
    problem = build_problem(
        load_domains({"air"}),
        "dirichlet",
        materials,
        [(zero, {"air": {"dirichlet"}}, False)],
    )

    solutions = problem.solve(preconditioner, q_mode)

    assert absolute_solution_norm(problem, solutions) <= 1e-10


@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_mixed_phi_circular_current(preconditioner, q_mode):
    materials = {
        "coil": {"coeff": MU0},
        "air": {
            "coeff": MU0,
            "gap": magnetic_flux,
        },
        "dirichlet": zero,
    }
    problem = build_problem(
        load_domains(),
        "dirichlet",
        materials,
        [
            (zero, {"coil": {"dirichlet"}, "air": {"dirichlet"}}, False),
            (analytic_H, {"air": {"gap"}}, True),
        ],
    )

    solutions = problem.solve(preconditioner, q_mode)

    assert absolute_solution_norm(problem, solutions) <= 1e-10
