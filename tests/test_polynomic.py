from pathlib import Path

import numpy as np
import pytest

from pyquasar import BemDomain, FemDomain, FetiProblem, load_mesh

TEST_GEO = Path(__file__).resolve().parents[1] / "examples" / "test.geo"
PRECONDITIONER_Q_CASES = [
    ("I", "I"),
    ("Dirichlet", "I"),
    ("Lumped Dirichlet", "I"),
    ("Dirichlet", "Diag"),
    ("Lumped Dirichlet", "Diag"),
    ("Dirichlet", "M"),
    ("Lumped Dirichlet", "M"),
]


def linear_u(points, normals):
    return 2 * points[..., 0] + 3 * points[..., 1] - 4


def linear_flow(points, normals):
    return 2 * normals[..., 0] + 3 * normals[..., 1]


def zero_source(points, normals):
    return 0


def quadratic_u(points, normals):
    return points[..., 0] ** 2


def quadratic_flow(points, normals):
    return 2 * points[..., 0] * normals[..., 0]


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def orthogonalize(vector, kernel):
    return vector - kernel * (vector @ kernel) / (kernel @ kernel)


def first_domain(domain_type):
    return next(domain_type(*data) for data in load_mesh(str(TEST_GEO)))


def build_feti_problem(use_bem, materials, refine=3):
    domain_type = BemDomain if use_bem else FemDomain
    domains = [domain_type(*data) for data in load_mesh(str(TEST_GEO), 10, refine)]
    problem = FetiProblem(domains)
    problem.assembly("dirichlet", materials)
    problem.add_skeleton_projection(
        materials["dirichlet"],
        {name: {"dirichlet"} for name in materials},
    )
    problem.decompose()
    return problem


def feti_solution_relative_error(problem, solutions, expected_func):
    actual_norm = 0
    error_norm = 0
    for domain, solution in zip(problem.domains, solutions):
        expected = expected_func(domain.vertices, 0)
        actual_norm += np.linalg.norm(expected) ** 2
        error_norm += np.linalg.norm(solution - expected) ** 2

    return (error_norm / actual_norm) ** 0.5


def assert_feti_solution_matches(problem, solutions, expected_func, tolerance):
    assert feti_solution_relative_error(problem, solutions, expected_func) <= tolerance


def test_fem_single_domain_linear_polynomial():
    domain = first_domain(FemDomain)
    dirichlet_vector = linear_u(domain.vertices[: domain.ext_dof_count], 0)

    domain.assembly({"dirichlet": linear_flow})
    solution = domain.solve_neumann(domain.load_vector)

    solution = orthogonalize(solution, domain.kernel[0])
    expected = orthogonalize(linear_u(domain.vertices, 0), domain.kernel[0])

    assert relative_error(solution, expected) <= 1e-8
    assert (
        relative_error(
            domain.solve_dirichlet(dirichlet_vector),
            domain.load_vector,
        )
        <= 1e-8
    )


def test_bem_single_domain_linear_polynomial():
    domain = first_domain(BemDomain)
    dirichlet_vector = linear_u(domain.vertices[: domain.ext_dof_count], 0)

    domain.assembly({"dirichlet": linear_flow})
    domain.decompose()
    solution = domain.solve_neumann(domain.load_vector)

    solution = orthogonalize(solution, domain.kernel[0])
    expected = orthogonalize(dirichlet_vector, domain.kernel[0])

    assert relative_error(solution, expected) <= 1e-4
    assert (
        relative_error(
            domain.solve_dirichlet(dirichlet_vector),
            domain.load_vector,
        )
        <= 1e-4
    )

    point_solution = domain.calc_solution(solution)
    constants = np.ones_like(point_solution)
    point_solution = orthogonalize(point_solution, constants)
    expected_points = orthogonalize(linear_u(domain.vertices, 0), constants)

    assert relative_error(point_solution, expected_points) <= 1e-4


@pytest.mark.parametrize(
    ("use_bem", "expected_func", "materials", "tolerance"),
    [
        pytest.param(
            False,
            linear_u,
            {
                "dirichlet": linear_u,
                "steel": {"neumann": linear_flow, "steel": zero_source},
            },
            1e-6,
            id="fem-linear",
        ),
        pytest.param(
            True,
            linear_u,
            {"dirichlet": linear_u, "steel": {"neumann": linear_flow, "steel": 0}},
            1e-5,
            id="bem-linear",
        ),
        pytest.param(
            False,
            quadratic_u,
            {
                "dirichlet": quadratic_u,
                "steel": {"neumann": quadratic_flow, "steel": -2},
            },
            5e-2,
            id="fem-quadratic",
        ),
        pytest.param(
            True,
            quadratic_u,
            {
                "dirichlet": quadratic_u,
                "steel": {"neumann": quadratic_flow, "steel": -2},
            },
            5e-2,
            id="bem-quadratic",
        ),
    ],
)
@pytest.mark.parametrize(
    ("preconditioner", "q_mode"),
    PRECONDITIONER_Q_CASES,
    ids=[
        f"{preconditioner}-{q_mode}"
        for preconditioner, q_mode in PRECONDITIONER_Q_CASES
    ],
)
def test_feti_polynomial_solution(
    use_bem,
    expected_func,
    materials,
    tolerance,
    preconditioner,
    q_mode,
):
    problem = build_feti_problem(use_bem, materials)

    solutions = problem.solve(preconditioner, q_mode)

    assert_feti_solution_matches(problem, solutions, expected_func, tolerance)


@pytest.mark.parametrize("use_bem", [False, True], ids=["fem", "bem"])
def test_quadratic_polynomial_refinement_convergence(use_bem):
    materials = {
        "dirichlet": quadratic_u,
        "steel": {"neumann": quadratic_flow, "steel": -2},
    }

    errors = []
    for refine in range(4):
        problem = build_feti_problem(use_bem, materials, refine)
        solutions = problem.solve("Dirichlet", "Diag")
        errors.append(feti_solution_relative_error(problem, solutions, quadratic_u))

    for coarse_error, refined_error in zip(errors, errors[1:]):
        assert refined_error < coarse_error
        assert coarse_error / refined_error >= 2.0
