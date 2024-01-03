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


def test_paper_figure1_upper_bound():
    expected = -0.421

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

    actual = identification(
        target="late",
        identified_estimands=["iv_slope"],
        basis_funcs=basis_funcs,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=u_part,
        u_lo_late_target=0.35,
        u_hi_late_target=0.9,
        u_lo_late_identified=None,
        u_hi_late_identified=None,
        support_z=DGP["support_z"],
        pscore_z=DGP["pscore_z"],
        pdf_z=DGP["pdf_z"],
        dz_cross=None,
        analytical_integration=False,
    )

    np.isclose(actual.fun, expected, atol=1e-4)
