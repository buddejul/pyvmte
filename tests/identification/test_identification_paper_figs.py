import pytest
from pyvmte.config import (
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    Estimand,
    Instrument,
    Setup,
)
from pyvmte.identification.identification import _compute_estimand, identification
from pyvmte.utilities import load_paper_dgp

DGP = load_paper_dgp()

INSTRUMENT = Instrument(
    support=DGP["support_z"],
    pmf=DGP["pdf_z"],
    pscores=DGP["pscores"],
)

U_PART = [0, 0.35, 0.6, 0.7, 0.9, 1]

BFUNC1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.35}
BFUNC2 = {"type": "constant", "u_lo": 0.35, "u_hi": 0.6}
BFUNC3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.7}
BFUNC4 = {"type": "constant", "u_lo": 0.7, "u_hi": 0.9}
BFUNC5 = {"type": "constant", "u_lo": 0.9, "u_hi": 1.0}

BASIS_FUNCS = [BFUNC1, BFUNC2, BFUNC3, BFUNC4, BFUNC5]


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

    estimand_late = Estimand(esttype="late", u_lo=0.35, u_hi=0.9)

    estimand_ols = Estimand(esttype="ols_slope")

    estimand_iv = Estimand(esttype="iv_slope")

    actual = [
        _compute(estimand) for estimand in [estimand_late, estimand_ols, estimand_iv]
    ]

    assert actual == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "setup",
    [(SETUP_FIG2), (SETUP_FIG3), (SETUP_FIG5)],
    ids=["fig2", "fig3", "fig5"],
)
def test_identification_paper_bounds(setup: Setup):
    expected = [setup.lower_bound, setup.upper_bound]

    target_estimand = setup.target
    identified_estimands = setup.identified_estimands

    result = identification(
        target=target_estimand,
        identified_estimands=identified_estimands,
        basis_funcs=BASIS_FUNCS,
        m0_dgp=DGP["m0"],
        m1_dgp=DGP["m1"],
        u_partition=U_PART,
        instrument=INSTRUMENT,
    )

    actual = [result["lower_bound"], result["upper_bound"]]
    assert actual == pytest.approx(expected, abs=1e-3)
