"""Identification kwargs for MST Figure 7."""

from pyvmte.config import (
    IV_MST,
    SETUP_FIG7,
    SETUP_SM_SHARP,
    U_PART_MST,
    _m0_paper,
    _m1_paper,
)
from pyvmte.utilities import generate_bernstein_basis_funcs

basis_funcs = generate_bernstein_basis_funcs(k=9)


KWARGS_ID = {
    "target": SETUP_FIG7.target,
    "identified_estimands": SETUP_FIG7.identified_estimands,
    "basis_funcs": basis_funcs,
    "m0_dgp": _m0_paper,
    "m1_dgp": _m1_paper,
    "instrument": IV_MST,
    "u_partition": U_PART_MST,
    "shape_constraints": SETUP_FIG7.shape_constraints,
    "mte_monotone": None,
    "monotone_response": None,
}


KWARGS_ID_SM_SHARP = {
    "target": SETUP_SM_SHARP.target,
    "identified_estimands": SETUP_SM_SHARP.identified_estimands,
}
