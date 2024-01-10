"""All the general configuration of the project."""
from pathlib import Path
import numpy as np

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

RNG = np.random.default_rng()

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]

SETUP_FIG2 = {
    "target": {"type": "late", "u_lo": 0.35, "u_hi": 0.9},
    "identified_estimands": {"type": "iv_slope"},
    "lower_bound": -0.421,
    "upper_bound": 0.500,
}

SETUP_FIG3 = {
    "target": {"type": "late", "u_lo": 0.35, "u_hi": 0.9},
    "identified_estimands": [
        {"type": "iv_slope"},
        {"type": "ols_slope"},
    ],
    "lower_bound": -0.411,
    "upper_bound": 0.500,
}

SETUP_FIG5 = {
    "target": {"type": "late", "u_lo": 0.35, "u_hi": 0.9},
    "identified_estimands": [
        {"type": "cross", "dz_cross": (0, 0)},
        {"type": "cross", "dz_cross": (0, 1)},
        {"type": "cross", "dz_cross": (0, 2)},
        {"type": "cross", "dz_cross": (1, 0)},
        {"type": "cross", "dz_cross": (1, 1)},
        {"type": "cross", "dz_cross": (1, 2)},
    ],
    "lower_bound": -0.138,
    "upper_bound": 0.407,
}


SETUP_MONTE_CARLO = {
    "sample_size": 10_000,
    "repetitions": 10_000,
}

SETUP_MONTE_CARLO_BY_TARGET = {
    "sample_size": 100,
    "repetitions": 100,
}
