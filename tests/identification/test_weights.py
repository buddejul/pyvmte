import numpy as np
import pandas as pd  # type: ignore
import pytest
from pyvmte.config import TEST_DIR
from pyvmte.identification.identification import _compute_choice_weights
from pyvmte.utilities import load_paper_dgp, _compute_constant_spline_weights
from pyvmte.estimation.estimation import _generate_basis_funcs, _create_funcs_from_dicts

DGP = load_paper_dgp()

INSTRUMENT = {
    "support_z": DGP["support_z"],
    "pdf_z": DGP["pdf_z"],
    "pscore_z": DGP["pscore_z"],
}

MOMENTS = {
    "expectation_d": DGP["expectation_d"],
    "variance_d": DGP["variance_d"],
    "expectation_z": DGP["expectation_z"],
    "covariance_dz": DGP["covariance_dz"],
}

BFUNC1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.35}
BFUNC2 = {"type": "constant", "u_lo": 0.35, "u_hi": 0.6}
BFUNC3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.7}
BFUNC4 = {"type": "constant", "u_lo": 0.7, "u_hi": 0.9}
BFUNC5 = {"type": "constant", "u_lo": 0.9, "u_hi": 1.0}

BFUNCS = [BFUNC1, BFUNC2, BFUNC3, BFUNC4, BFUNC5]

BFUNC_LENS = np.array([bfunc["u_hi"] - bfunc["u_lo"] for bfunc in BFUNCS])


def test_compute_choice_weights_late():
    target = {"type": "late", "u_lo": 0.35, "u_hi": 0.9}

    weight = 1 / (target["u_hi"] - target["u_lo"])
    expected = np.array(
        [0, -weight, -weight, -weight, 0, 0, weight, weight, weight, 0]
    ) * np.tile(BFUNC_LENS, 2)

    actual = _compute_choice_weights(
        target=target, basis_funcs=BFUNCS, instrument=INSTRUMENT
    )

    assert actual == pytest.approx(expected)


def test_compute_constant_spline_weights_ols_slope_d0():
    # Calculated myself but looks very close to weights plotted in MST 2018 ECMA Fig 3
    weights = [
        0.0,
        -0.970873786407767,
        -1.7475728155339807,
        -1.94174757281553,
        -1.94174757281553,
    ]
    expected = np.array(weights) * BFUNC_LENS
    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "ols_slope"},
            basis_function=bfunc,
            d=0,
            moments=MOMENTS,
            instrument=INSTRUMENT,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)


expected = [2.061855670103093, 1.0309278350515467, 0.20618556701030932, 0.0]


def test_compute_constant_spline_weights_ols_slope_d1():
    # Calculated myself but looks very close to weights plotted in MST 2018 ECMA Fig 3
    weights = [2.061855670103093, 1.0309278350515467, 0.20618556701030932, 0, 0]
    expected = np.array(weights) * BFUNC_LENS

    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "ols_slope"},
            basis_function=bfunc,
            d=1,
            moments=MOMENTS,
            instrument=INSTRUMENT,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)


def test_compute_constant_spline_weights_iv_slope_d0():
    # Calculated myself but looks very close to weights plotted in MST 2018 ECMA Fig 3
    weights = [0.0, -3.3707865168539333, -1.5730337078651693, 0, 0]
    expected = np.array(weights) * BFUNC_LENS

    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "iv_slope"},
            basis_function=bfunc,
            d=0,
            moments=MOMENTS,
            instrument=INSTRUMENT,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)


def test_compute_constant_spline_weights_iv_slope_d1():
    # Calculated myself but looks very close to weights plotted in MST 2018 ECMA Fig 3
    weights = [0, 3.370786516853933, 1.5730337078651686, 0, 0.0]
    expected = np.array(weights) * BFUNC_LENS

    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "iv_slope"},
            basis_function=bfunc,
            d=1,
            moments=MOMENTS,
            instrument=INSTRUMENT,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)


def test_compute_constant_spline_weights_late_d0():
    u_lo = 0.35
    u_hi = 0.9

    weight = 1 / (u_hi - u_lo)

    expected = np.array([0, -weight, -weight, -weight, 0]) * BFUNC_LENS

    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "late", "u_lo": u_lo, "u_hi": u_hi},
            basis_function=bfunc,
            d=0,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)


def test_compute_constant_spline_weights_late_d1():
    u_lo = 0.35
    u_hi = 0.9

    points_to_evaluate = [0.1, 0.36, 0.62, 0.72, 0.95]

    weight = 1 / (u_hi - u_lo)

    expected = np.array([0, weight, weight, weight, 0]) * BFUNC_LENS

    actual = []

    for bfunc in BFUNCS:
        result = _compute_constant_spline_weights(
            estimand={"type": "late", "u_lo": u_lo, "u_hi": u_hi},
            basis_function=bfunc,
            d=1,
        )
        actual.append(result)

    assert actual == pytest.approx(expected)
