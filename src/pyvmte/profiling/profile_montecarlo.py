"""Setup for profiling monte carlo functions."""
import numpy as np

from pyvmte.config import SETUP_FIG5

rng = np.random.default_rng()

target = SETUP_FIG5.target
identified_estimands = SETUP_FIG5.identified_estimands
