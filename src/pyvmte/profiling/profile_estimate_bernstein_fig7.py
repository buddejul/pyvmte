"""Profiling for estimation with bernstein polynomials."""
from pyvmte.config import RNG, SETUP_FIG7
from pyvmte.utilities import simulate_data_from_paper_dgp

setup = SETUP_FIG7
k_degree = 10

basis_func_options = {"k_degree": k_degree}
basis_func_type = "bernstein"
shape_constraints = setup.shape_constraints

data = simulate_data_from_paper_dgp(sample_size=10_000, rng=RNG)

_kwargs = {
    "target": setup.target,
    "identified_estimands": setup.identified_estimands,
    "basis_func_type": basis_func_type,
    "y_data": data["y"],
    "d_data": data["d"],
    "z_data": data["z"],
    "shape_constraints": shape_constraints,
    "basis_func_options": basis_func_options,
    "method": "highs",
}
