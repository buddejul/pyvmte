"""Test computation of identification results from MST."""
import pytest
from pyvmte.classes import Estimand, Setup
from pyvmte.config import (
    BFUNCS_MST,
    DGP_MST,
    IV_MST,
    PARAMS_MST,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    U_PART_MST,
)
from pyvmte.identification.identification import _compute_estimand, identification


@pytest.mark.parametrize(
    ("target", "kwargs", "expected"),
    [
        ("ols_slope", {}, PARAMS_MST["ols_slope"]),
        ("iv_slope", {}, PARAMS_MST["iv_slope"]),
        ("late", {"u_lo": 0.35, "u_hi": 0.9}, PARAMS_MST["late"]),
    ],
)
def test_compute_estimand(target, kwargs, expected):
    target = Estimand(esttype=target, **kwargs)

    actual = _compute_estimand(
        estimand=target,
        m0=DGP_MST.m0,
        m1=DGP_MST.m1,
        u_part=U_PART_MST,
        instrument=IV_MST,
    )

    assert actual == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    ("setup", "method"),
    [
        (SETUP_FIG2, "highs"),
        (SETUP_FIG3, "highs"),
        (SETUP_FIG5, "highs"),
        (SETUP_FIG2, "copt"),
        (SETUP_FIG3, "copt"),
        (SETUP_FIG5, "copt"),
    ],
    ids=[
        "fig2_highs",
        "fig3_highs",
        "fig5_highs",
        "fig2_copt",
        "fig3_copt",
        "fig5_copt",
    ],
)
def test_identification_paper_bounds(setup: Setup, method: str):
    expected = [setup.lower_bound, setup.upper_bound]

    target_estimand = setup.target
    identified_estimands = setup.identified_estimands

    result = identification(
        target=target_estimand,
        identified_estimands=identified_estimands,
        basis_funcs=BFUNCS_MST,
        m0_dgp=DGP_MST.m0,
        m1_dgp=DGP_MST.m1,
        u_partition=U_PART_MST,
        instrument=IV_MST,
        method=method,
    )

    actual = [result["lower_bound"], result["upper_bound"]]
    assert actual == pytest.approx(expected, abs=1e-3)
