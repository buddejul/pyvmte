import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR
from pyvmte.identification.identification import _compute_estimand, identification
from pyvmte.utilities import load_paper_dgp

from pyvmte.utilities import bern_bas

DGP = load_paper_dgp()


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
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_ols_slope():
    expected = DGP["ols_slope"]

    target_estimand = {
        "type": "ols_slope",
    }

    actual = _compute_estimand(
        estimand=target_estimand,
        m0=DGP["m0"],
        m1=DGP["m1"],
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_iv_slope():
    expected = DGP["iv_slope"]

    target_estimand = {"type": "iv_slope"}

    actual = _compute_estimand(
        estimand=target_estimand,
        m0=DGP["m0"],
        m1=DGP["m1"],
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
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
            support_z=DGP["support_z"],
            pscore_z=DGP["pscore_z"],
            pdf_z=DGP["pdf_z"],
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


def test_paper_figure1_upper_bound():
    expected = [-0.421, 0.500]

    u_part = [0, 0.35, 0.6, 0.7, 0.9, 1]

    for i, (low, high) in enumerate(zip(u_part[:-1], u_part[1:]), start=1):
        globals()[f"bern_bas_{i}"] = (
            lambda x, low=low, high=high: bern_bas(2, 0, x)
            + bern_bas(2, 1, x)
            + bern_bas(2, 2, x)
            if low <= x < high
            else 0
        )

    basis_funcs = [bern_bas_1, bern_bas_2, bern_bas_3, bern_bas_4, bern_bas_5]

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
        basis_funcs=basis_funcs,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=u_part,
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
        analytical_integration=False,
    )

    np.isclose(list(actual.values()), expected, atol=1e-4)
