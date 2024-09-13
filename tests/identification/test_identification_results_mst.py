"""Test computation of identification results from MST."""
import pytest
from pyvmte.classes import Estimand, Setup
from pyvmte.config import (
    BFUNCS_MST,
    DGP_MST,
    IV_MST,
    PARAMS_MST,
    SETUP_FIG7,
    U_PART_MST,
)
from pyvmte.identification.identification import _compute_estimand, identification
from pyvmte.utilities import generate_bernstein_basis_funcs


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
        (SETUP_FIG7, "highs"),
        (SETUP_FIG7, "copt"),
    ],
    ids=[
        # "fig2_highs",
        # "fig3_highs",
        # "fig5_highs",
        # "fig6_highs",
        "fig7_highs",
        # "fig2_copt",
        # "fig3_copt",
        # "fig5_copt",
        # "fig6_copt",
        "fig7_copt",
    ],
)
def test_identification_paper_bounds(setup: Setup, method: str):
    expected = [setup.lower_bound, setup.upper_bound]

    target_estimand = setup.target
    identified_estimands = setup.identified_estimands

    if setup.polynomial is not None and setup.polynomial[0] == "bernstein":
        bfuncs = generate_bernstein_basis_funcs(k=setup.polynomial[1])
    else:
        bfuncs = BFUNCS_MST

    result = identification(
        target=target_estimand,
        identified_estimands=identified_estimands,
        basis_funcs=bfuncs,
        m0_dgp=DGP_MST.m0,
        m1_dgp=DGP_MST.m1,
        u_partition=U_PART_MST,
        instrument=IV_MST,
        method=method,
        shape_constraints=setup.shape_constraints,
    )

    actual = [result["lower_bound"], result["upper_bound"]]
    assert actual == pytest.approx(expected, abs=1e-3)
