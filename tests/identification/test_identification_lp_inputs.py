import numpy as np
import pandas as pd  # type: ignore
import pytest
from pyvmte.config import TEST_DIR, SETUP_FIG3, Instrument

from pyvmte.identification.identification import (
    _compute_choice_weights,
)

from pyvmte.utilities import load_paper_dgp

from pyvmte.utilities import bern_bas

from itertools import product

DGP = load_paper_dgp()

INSTRUMENT = Instrument(
    support=DGP["support_z"],
    pmf=DGP["pdf_z"],
    pscores=DGP["pscore_z"],
)

U_PART = [0, 0.35, 0.6, 0.7, 0.9, 1]

BFUNC1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.35}
BFUNC2 = {"type": "constant", "u_lo": 0.35, "u_hi": 0.6}
BFUNC3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.7}
BFUNC4 = {"type": "constant", "u_lo": 0.7, "u_hi": 0.9}
BFUNC5 = {"type": "constant", "u_lo": 0.9, "u_hi": 1.0}

BASIS_FUNCS = [BFUNC1, BFUNC2, BFUNC3, BFUNC4, BFUNC5]


def test_lp_input_c_figure_3():
    target = SETUP_FIG3.target
    late_weight = 1 / (target.u_hi - target.u_lo)
    expected = [
        0,
        -late_weight * (BFUNC2["u_hi"] - BFUNC2["u_lo"]),
        -late_weight * (BFUNC3["u_hi"] - BFUNC3["u_lo"]),
        -late_weight * (BFUNC4["u_hi"] - BFUNC4["u_lo"]),
        0,
        0,
        late_weight * (BFUNC2["u_hi"] - BFUNC2["u_lo"]),
        late_weight * (BFUNC3["u_hi"] - BFUNC3["u_lo"]),
        late_weight * (BFUNC4["u_hi"] - BFUNC4["u_lo"]),
        0,
    ]

    actual = _compute_choice_weights(
        target=target,
        basis_funcs=BASIS_FUNCS,
        instrument=INSTRUMENT,
    )

    assert expected == pytest.approx(actual)


def test_lp_input_A_eq_figure_3():
    pass


def test_lp_input_b_eq_figure_3():
    pass
