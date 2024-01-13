import numpy as np
import pandas as pd  # type: ignore
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

BFUNC1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.35}
BFUNC2 = {"type": "constant", "u_lo": 0.35, "u_hi": 0.6}
BFUNC3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.7}
BFUNC4 = {"type": "constant", "u_lo": 0.7, "u_hi": 0.9}
BFUNC5 = {"type": "constant", "u_lo": 0.9, "u_hi": 1.0}

BASIS_FUNCS = [BFUNC1, BFUNC2, BFUNC3, BFUNC4, BFUNC5]


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

    assert actual == pytest.approx(expected, abs=1e-4)


def test_paper_ols_slope():
    expected = DGP["ols_slope"]

    target_estimand = {
        "type": "ols_slope",
    }

    actual = _compute_estimand(
        estimand=target_estimand, m0=DGP["m0"], m1=DGP["m1"], instrument=INSTRUMENT
    )

    assert actual == pytest.approx(expected, abs=1e-4)


def test_paper_iv_slope():
    expected = DGP["iv_slope"]

    target_estimand = {"type": "iv_slope"}

    actual = _compute_estimand(
        estimand=target_estimand, m0=DGP["m0"], m1=DGP["m1"], instrument=INSTRUMENT
    )

    assert actual == pytest.approx(expected, abs=1e-3)


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

    assert actual == pytest.approx(expected, abs=1e-3)


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

    actual = [actual["lower_bound"], actual["upper_bound"]]
    assert actual == pytest.approx(expected, abs=1e-3)


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

    actual = [actual["lower_bound"], actual["upper_bound"]]
    assert actual == pytest.approx(expected, abs=1e-3)


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
    bounds = identification(
        target=target_estimand,
        identified_estimands=cross_estimands,
        basis_funcs=BASIS_FUNCS,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=U_PART,
        instrument=INSTRUMENT,
        analytical_integration=False,
    )

    actual = [bounds["lower_bound"], bounds["upper_bound"]]

    assert actual == pytest.approx(expected, abs=1e-3)
