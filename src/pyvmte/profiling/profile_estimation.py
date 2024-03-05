"""Setup for profiling estimation functions."""
import numpy as np

from pyvmte.config import RNG, SETUP_FIG3
from pyvmte.utilities import simulate_data_from_paper_dgp

setup = SETUP_FIG3

SAMPLE_SIZE = 100_000

data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)

y_data = np.array(data["y"])
d_data = np.array(data["d"])
z_data = np.array(data["z"])

target = setup.target
identified_estimands = setup.identified_estimands

basis_func_type = "constant"
