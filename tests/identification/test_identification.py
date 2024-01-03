import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR
from pyvmte.identification.identification import _compute_estimand, identification
from pyvmte.utilities import load_paper_dgp

from pyvmte.utilities import bern_bas

from itertools import product

DGP = load_paper_dgp()

INSTRUMENT = {
    "support_z": DGP["support_z"],
    "pscore_z": DGP["pscore_z"],
    "pdf_z": DGP["pdf_z"],
}

U_PART = [0, 0.35, 0.6, 0.7, 0.9, 1]

for i, (low, high) in enumerate(zip(U_PART[:-1], U_PART[1:]), start=1):
    globals()[f"bern_bas_{i}"] = (
        lambda x, low=low, high=high: bern_bas(2, 0, x)
        + bern_bas(2, 1, x)
        + bern_bas(2, 2, x)
        if low <= x < high
        else 0
    )

BASIS_FUNCS = [bern_bas_1, bern_bas_2, bern_bas_3, bern_bas_4, bern_bas_5]


def test_paper_late():
    expected = DGP["late_35_90"]

    target_estimand = {
        "type": "late",
        "u_lo": 0.35,
        "u_hi": 0.9,
    }

    actual = _compute_estimand(
        estimand=target_estimand,
        m0=DGP["m0"],
        m1=DGP["m1"],
        u_part=[0.35, 0.9],
        instrument=INSTRUMENT,
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_ols_slope():
    expected = DGP["ols_slope"]

    target_estimand = {
        "type": "ols_slope",
    }

    actual = _compute_estimand(
        estimand=target_estimand, m0=DGP["m0"], m1=DGP["m1"], instrument=INSTRUMENT
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_iv_slope():
    expected = DGP["iv_slope"]

    target_estimand = {"type": "iv_slope"}

    actual = _compute_estimand(
        estimand=target_estimand, m0=DGP["m0"], m1=DGP["m1"], instrument=INSTRUMENT
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_late_ols_iv():
    expected = [DGP["late_35_90"], DGP["ols_slope"], DGP["iv_slope"]]

    def _compute(estimand):
        return _compute_estimand(
            estimand=estimand,
            m0=DGP["m0"],
            m1=DGP["m1"],
            u_part=[0.35, 0.9],
            instrument=INSTRUMENT,
        )

    estimand_late = {
        "type": "late",
        "u_lo": 0.35,
        "u_hi": 0.9,
    }

    estimand_ols = {
        "type": "ols_slope",
    }

    estimand_iv = {
        "type": "iv_slope",
    }

    actual = [
        _compute(estimand) for estimand in [estimand_late, estimand_ols, estimand_iv]
    ]

    # Check that all values are close to expected in one statement
    np.allclose(actual, expected, atol=1e-4)


def test_paper_figure2_bounds():
    expected = [-0.421, 0.500]

    target_estimand = {
        "type": "late",
        "u_lo": 0.35,
        "u_hi": 0.9,
    }

    iv_estimand = {
        "type": "iv_slope",
    }

    actual = identification(
        target=target_estimand,
        identified_estimands=iv_estimand,
        basis_funcs=BASIS_FUNCS,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=U_PART,
        instrument=INSTRUMENT,
        analytical_integration=False,
    )

    np.isclose(list(actual.values()), expected, atol=1e-4)


def test_paper_figure3_bounds():
    expected = [-0.411, 0.500]

    target_estimand = {
        "type": "late",
        "u_lo": 0.35,
        "u_hi": 0.9,
    }

    iv_estimand = {
        "type": "iv_slope",
    }

    ols_estimand = {
        "type": "ols_slope",
    }

    actual = identification(
        target=target_estimand,
        identified_estimands=[iv_estimand, ols_estimand],
        basis_funcs=BASIS_FUNCS,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=U_PART,
        instrument=INSTRUMENT,
        analytical_integration=False,
    )

    np.isclose(list(actual.values()), expected, atol=1e-4)


def test_paper_figure5_bounds():
    expected = [-0.138, 0.407]

    target_estimand = {
        "type": "late",
        "u_lo": 0.35,
        "u_hi": 0.9,
    }

    combinations = product([0, 1], [0, 1, 2])

    cross_estimands = [
        {"type": "cross", "dz_cross": list(comb)} for comb in combinations
    ]

    actual = identification(
        target=target_estimand,
        identified_estimands=cross_estimands,
        basis_funcs=BASIS_FUNCS,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=U_PART,
        instrument=INSTRUMENT,
        analytical_integration=False,
    )

    np.isclose(list(actual.values()), expected, atol=1e-4)
