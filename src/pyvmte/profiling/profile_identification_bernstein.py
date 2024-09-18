"""Profiling for identification with bernstein polynomials."""

# Old time: In [11]: %timeit identification(**_kwargs)
# 1.07 s ± 9.93 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# After removing 2d-quad:
# 290 ms ± 6.27 ms per loop


from pyvmte.config import DGP_MST, IV_MST, SETUP_FIG7, U_PART_MST
from pyvmte.identification import identification  # noqa: F401
from pyvmte.utilities import generate_bernstein_basis_funcs

setup = SETUP_FIG7

bfuncs = generate_bernstein_basis_funcs(k=10)

_kwargs = {
    "target": setup.target,
    "identified_estimands": setup.identified_estimands,
    "basis_funcs": bfuncs,
    "m0_dgp": DGP_MST.m0,
    "m1_dgp": DGP_MST.m1,
    "u_partition": U_PART_MST,
    "instrument": IV_MST,
    "shape_constraints": setup.shape_constraints,
}
