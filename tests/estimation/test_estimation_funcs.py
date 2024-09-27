import numpy as np
import pytest
from pyvmte.config import BFUNCS_MST, RNG, U_PART_MST, Estimand
from pyvmte.estimation.estimation import (
    _build_first_step_ub_matrix,
    _compute_first_step_bounds,
    _compute_first_step_upper_bounds,
    _compute_second_step_bounds,
    _compute_second_step_upper_bounds,
    _compute_u_partition,
    _estimate_identified_estimands,
    _estimate_instrument_characteristics,
    _estimate_moments_for_weights,
    _estimate_prop_z,
    _estimate_weights_estimand,
    _first_step_linear_program,
    _generate_array_of_pscores,
    _generate_basis_funcs,
    _second_step_linear_program,
)
from pyvmte.utilities import simulate_data_from_paper_dgp


def test_generate_basis_funcs():
    actual = _generate_basis_funcs(basis_func_type="constant", u_partition=U_PART_MST)
    assert actual == BFUNCS_MST


@pytest.fixture()
def data():
    return {
        "d": np.array([0, 1, 0, 1, 1, 0, 1, 1, 1]),
        "z": np.array([1, 1, 2, 2, 2, 3, 3, 3, 3]),
        "pscores": np.array([0.5, 2 / 3, 0.75]),
    }


def test_estimate_prop_z(data):
    expected = np.array([0.5, 2 / 3, 0.75])

    actual = _estimate_prop_z(data["z"], data["d"], support=np.unique(data["z"]))

    assert actual == pytest.approx(expected)


def test_generate_array_of_pscores(data):
    expected = np.array([0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 0.75, 0.75, 0.75, 0.75])

    actual = _generate_array_of_pscores(
        data["z"],
        np.unique(data["z"]),
        data["pscores"],
    )

    assert actual == pytest.approx(expected)


@pytest.fixture()
def setup_lp_inputs():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    data = {
        "d": RNG.choice([0, 1], size=10_000),
        "z": RNG.choice([1, 2, 3], size=10_000),
    }

    instrument = _estimate_instrument_characteristics(data["z"], data["d"])

    data["pscores"] = _generate_array_of_pscores(
        data["z"],
        instrument.support,
        instrument.pscores,
    )

    identified_estimands = [Estimand(esttype="iv_slope"), Estimand(esttype="ols_slope")]
    beta_hat = RNG.normal(size=len(identified_estimands))

    return {
        "u_partition": u_partition,
        "basis_funcs": basis_funcs,
        "data": data,
        "instrument": instrument,
        "identified_estimands": identified_estimands,
        "beta_hat": beta_hat,
    }


def test_build_first_step_ub_matrix_shape(setup_lp_inputs):
    a_ub = _build_first_step_ub_matrix(
        basis_funcs=setup_lp_inputs["basis_funcs"],
        identified_estimands=setup_lp_inputs["identified_estimands"],
        data=setup_lp_inputs["data"],
        instrument=setup_lp_inputs["instrument"],
    )

    expected = (2 * 2, 4 * 2 + 2)
    actual = a_ub.shape

    assert actual == expected


def test_compute_first_step_bounds(setup_lp_inputs):
    expected = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (None, None),
        (None, None),
    ]

    actual = _compute_first_step_bounds(
        setup_lp_inputs["identified_estimands"],
        setup_lp_inputs["basis_funcs"],
    )
    assert actual == pytest.approx(expected)


def test_first_step_linear_program_runs_and_non_zero(setup_lp_inputs):
    result = _first_step_linear_program(
        identified_estimands=setup_lp_inputs["identified_estimands"],
        basis_funcs=setup_lp_inputs["basis_funcs"],
        data=setup_lp_inputs["data"],
        beta_hat=setup_lp_inputs["beta_hat"],
        instrument=setup_lp_inputs["instrument"],
        method="highs",
        shape_constraints=None,
    )

    assert result["minimal_deviations"] != 0


def test_compute_second_step_bounds(setup_lp_inputs):
    actual = _compute_second_step_bounds(
        setup_lp_inputs["basis_funcs"],
        setup_lp_inputs["identified_estimands"],
    )

    expected = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (None, None),
        (None, None),
    ]
    assert actual == pytest.approx(expected)


def test_compute_first_step_upper_bounds():
    beta_hat = np.array([1, 2, 3])
    expected = np.array([1, 2, 3, -1, -2, -3])

    actual = _compute_first_step_upper_bounds(beta_hat)
    assert actual == pytest.approx(expected)


@pytest.fixture()
def setup_dgp():
    identified_estimands = [Estimand(esttype="iv_slope"), Estimand(esttype="ols_slope")]
    target = Estimand(esttype="late", u_lo=0.35, u_hi=0.9)
    sample_size = 100_000
    data = simulate_data_from_paper_dgp(sample_size=sample_size, rng=RNG)
    instrument = _estimate_instrument_characteristics(data["z"], data["d"])
    u_partition = _compute_u_partition(target, instrument.pscores)
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    data["pscores"] = _generate_array_of_pscores(
        data["z"],
        instrument.support,
        instrument.pscores,
    )

    beta_hat = _estimate_identified_estimands(
        identified_estimands=identified_estimands,
        y_data=data["y"],
        d_data=data["d"],
        z_data=data["z"],
    )

    moments = _estimate_moments_for_weights(z_data=data["z"], d_data=data["d"])

    return {
        "identified_estimands": identified_estimands,
        "target": target,
        "data": data,
        "instrument": instrument,
        "u_partition": u_partition,
        "basis_funcs": basis_funcs,
        "beta_hat": beta_hat,
        "tolerance": 1 / sample_size,
        "minimal_deviations": 10e-5,
        "moments": moments,
    }


def test_second_step_linear_program_runs(setup_dgp):
    kwargs_names = [
        "target",
        "identified_estimands",
        "basis_funcs",
        "data",
        "minimal_deviations",
        "tolerance",
        "beta_hat",
        "instrument",
    ]
    kwargs = {name: setup_dgp[name] for name in kwargs_names}

    result = _second_step_linear_program(
        **kwargs,
        method="highs",
        shape_constraints=None,
    )

    assert result is not None


def test_compute_second_step_upper_bounds(setup_dgp):
    beta_hat = np.array([1, 2, 3])

    expected = np.concatenate(
        [
            np.array([setup_dgp["minimal_deviations"] + setup_dgp["tolerance"]]),
            beta_hat,
            -beta_hat,
        ],
    )

    actual = _compute_second_step_upper_bounds(
        minimal_deviations=setup_dgp["minimal_deviations"],
        tolerance=setup_dgp["tolerance"],
        beta_hat=beta_hat,
    )

    assert actual == pytest.approx(expected)


def test_compute_u_partition():
    target = Estimand(esttype="late", u_lo=0.35, u_hi=0.9)
    pscore_z = [0.1, 0.2, 0.64, 0.83]

    expected = [0, 0.1, 0.2, 0.35, 0.64, 0.83, 0.9, 1]

    actual = _compute_u_partition(target=target, pscore_z=pscore_z)

    assert actual == pytest.approx(expected)


def test_estimate_weights_estimand_symmetry(setup_dgp):
    actual = _estimate_weights_estimand(
        estimand=Estimand(esttype="iv_slope"),
        basis_funcs=setup_dgp["basis_funcs"],
        data=setup_dgp["data"],
        instrument=setup_dgp["instrument"],
        moments=setup_dgp["moments"],
    )

    first_half = np.array(actual[: len(actual) // 2])
    second_half = -1 * np.array(actual[len(actual) // 2 :])

    assert first_half == pytest.approx(second_half)


def test_build_first_step_ub_matrix_symmetry(setup_dgp):
    a_ub = _build_first_step_ub_matrix(
        basis_funcs=setup_dgp["basis_funcs"],
        identified_estimands=setup_dgp["identified_estimands"],
        data=setup_dgp["data"],
        instrument=setup_dgp["instrument"],
    )

    num_bfuncs = len(setup_dgp["basis_funcs"]) * 2
    weights = a_ub[0, :num_bfuncs]

    assert weights[: num_bfuncs // 2] == pytest.approx(-1 * weights[num_bfuncs // 2 :])
