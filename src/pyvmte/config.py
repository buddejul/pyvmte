"""All the general configuration of the project."""

from itertools import product
from pathlib import Path

import numpy as np

from pyvmte.classes import DGP, Estimand, Instrument, Setup
from pyvmte.utilities import bern_bas

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

SIMULATION_RESULTS_DIR = BLD.joinpath("python", "data").resolve()

RNG = np.random.default_rng()

__all__ = ["BLD", "SRC", "TEST_DIR"]

SETUP_FIG2 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=[Estimand(esttype="iv_slope")],
    lower_bound=-0.421,
    upper_bound=0.500,
)

SETUP_FIG3 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=[
        Estimand(esttype="iv_slope"),
        Estimand(esttype="ols_slope"),
    ],
    lower_bound=-0.411,
    upper_bound=0.500,
)

combinations = product([0, 1], [0, 1, 2])

cross_estimands = [
    Estimand(esttype="cross", dz_cross=tuple(comb)) for comb in combinations  # type: ignore
]

SETUP_FIG5 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=cross_estimands,
    lower_bound=-0.138,
    upper_bound=0.407,
)

SETUP_FIG6 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=cross_estimands,
    lower_bound=-0.095,
    upper_bound=0.077,
    shape_constraints=("decreasing", "decreasing"),
)

SETUP_FIG7 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=cross_estimands,
    lower_bound=-0.000,
    upper_bound=0.067,
    shape_constraints=("decreasing", "decreasing"),
    polynomial=("bernstein", 9),
)

IV_MST = Instrument(
    support=np.array([0, 1, 2]),
    pmf=np.array([0.5, 0.4, 0.1]),
    pscores=np.array([0.35, 0.6, 0.7]),
)

bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.35}
bfunc_2 = {"type": "constant", "u_lo": 0.35, "u_hi": 0.6}
bfunc_3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.7}
bfunc_4 = {"type": "constant", "u_lo": 0.7, "u_hi": 0.9}
bfunc_5 = {"type": "constant", "u_lo": 0.9, "u_hi": 1.0}

BFUNCS_MST = [bfunc_1, bfunc_2, bfunc_3, bfunc_4, bfunc_5]

BFUNC_LENS_MST = np.array([bfunc["u_hi"] - bfunc["u_lo"] for bfunc in BFUNCS_MST])  # type: ignore


def _m0_paper(u: float) -> float:
    return 0.6 * bern_bas(2, 0, u) + 0.4 * bern_bas(2, 1, u) + 0.3 * bern_bas(2, 2, u)


def _m1_paper(u: float) -> float:
    return 0.75 * bern_bas(2, 0, u) + 0.5 * bern_bas(2, 1, u) + 0.25 * bern_bas(2, 2, u)


DGP_MST = DGP(
    m0=_m0_paper,
    m1=_m1_paper,
    support_z=np.array([0, 1, 2]),
    pmf_z=np.array([0.5, 0.4, 0.1]),
    pscores=np.array([0.35, 0.6, 0.7]),
    joint_pmf_dz={
        1: {0: 0.175, 1: 0.24, 2: 0.07},
        0: {0: 0.325, 1: 0.16, 2: 0.03},
    },
)


U_PART_MST = np.array([0, 0.35, 0.6, 0.7, 0.9, 1])

PARAMS_MST = {
    "ols_slope": 0.253,
    "late": 0.046,
    "iv_slope": 0.074,
}


SETUP_SM_IDLATE = Setup(
    target=Estimand(esttype="late"),
    identified_estimands=[Estimand(esttype="late", u_lo=0.4, u_hi=0.6)],
    lower_bound=np.nan,
    upper_bound=np.nan,
)

SETUP_SM_SHARP = Setup(
    target=Estimand(esttype="late"),
    identified_estimands=[
        Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
    ],
    lower_bound=np.nan,
    upper_bound=np.nan,
)
