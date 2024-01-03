import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR
from pyvmte.identification.identification import _compute_estimand
from pyvmte.utilities import load_paper_dgp

DGP = load_paper_dgp()


def test_paper_late():
    expected = DGP["late_35_90"]

    actual = _compute_estimand(
        estimand="late",
        m0=DGP["m0"],
        m1=DGP["m1"],
        u_lo=0.35,
        u_hi=0.9,
        u_part=[0.35, 0.9],
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_ols_slope():
    expected = DGP["ols_slope"]

    actual = _compute_estimand(
        estimand="ols_slope",
        m0=DGP["m0"],
        m1=DGP["m1"],
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
    )

    np.isclose(actual, expected, atol=1e-4)


def test_paper_iv_slope():
    expected = DGP["iv_slope"]

    actual = _compute_estimand(
        estimand="iv_slope",
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
            u_lo=0.35,
            u_hi=0.9,
            u_part=[0.35, 0.9],
            support_z=DGP["support_z"],
            pscore_z=DGP["pscore_z"],
            pdf_z=DGP["pdf_z"],
        )

    actual = [_compute(estimand) for estimand in ["late", "ols_slope", "iv_slope"]]

    # Check that all values are close to expected in one statement
    np.allclose(actual, expected, atol=1e-4)
