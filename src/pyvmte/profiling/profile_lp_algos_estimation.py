"""Setup profiling for comparing LP algos."""

from pyvmte.config import RNG, SETUP_FIG3
from pyvmte.utilities import simulate_data_from_paper_dgp

target = SETUP_FIG3.target
identified_estimands = SETUP_FIG3.identified_estimands
basis_func_type = "constant"

data = simulate_data_from_paper_dgp(10_000, RNG)
y_data = data["y"]
d_data = data["d"]
z_data = data["z"]
