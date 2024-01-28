"""All the general configuration of the project."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np


@dataclass
class Estimand:
    esttype: str
    u_lo: float | None = None
    u_hi: float | None = None
    dz_cross: tuple[int, int] | None = None


@dataclass
class Instrument:
    support: np.ndarray
    pmf: np.ndarray
    pscores: np.ndarray


class Setup(NamedTuple):
    target: Estimand
    identified_estimands: list[Estimand]
    lower_bound: float
    upper_bound: float


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
    Estimand(esttype="cross", dz_cross=tuple(comb)) for comb in combinations
]

SETUP_FIG5 = Setup(
    target=Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
    identified_estimands=cross_estimands,
    lower_bound=-0.138,
    upper_bound=0.407,
)


SETUP_MONTE_CARLO = {
    "sample_size": 10_000,
    "repetitions": 10_000,
}

SETUP_MONTE_CARLO_BY_TARGET = {
    "sample_size": 10_000,
    "repetitions": 1_000,
}

U_HI_RANGE = np.arange(0.35, 1, 0.05)

IV_PAPER = Instrument(
    support=np.array([0, 1, 2]),
    pmf=np.array([0.5, 0.4, 0.1]),
    pscores=np.array([0.35, 0.6, 0.7]),
)
