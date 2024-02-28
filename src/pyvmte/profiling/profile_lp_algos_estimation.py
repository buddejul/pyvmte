"""Setup profiling for comparing LP algos."""
import numpy as np

from pyvmte.config import SETUP_FIG3
from pyvmte.utilities import simulate_data_from_paper_dgp

target = SETUP_FIG3.target
identified_estimands = SETUP_FIG3.identified_estimands
basis_func_type = "constant"

rng = np.random.default_rng()

data = simulate_data_from_paper_dgp(10_000, rng)
y_data = data["y"]
d_data = data["d"]
z_data = data["z"]
