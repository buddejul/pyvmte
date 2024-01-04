import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR

from pyvmte.estimation.estimation import _generate_basis_funcs


def test_generate_basis_funcs():
    u_partition = [0, 0.35, 0.65, 0.7, 1]

    expected = [
        {"d_value": 0, "u_lo": 0, "u_hi": 0.35},
        {"d_value": 0, "u_lo": 0.35, "u_hi": 0.65},
        {"d_value": 0, "u_lo": 0.65, "u_hi": 0.7},
        {"d_value": 0, "u_lo": 0.7, "u_hi": 1},
        {"d_value": 1, "u_lo": 0, "u_hi": 0.35},
        {"d_value": 1, "u_lo": 0.35, "u_hi": 0.65},
        {"d_value": 1, "u_lo": 0.65, "u_hi": 0.7},
        {"d_value": 1, "u_lo": 0.7, "u_hi": 1},
    ]

    actual = _generate_basis_funcs(basis_func_type="constant", u_partition=u_partition)

    assert actual == expected
