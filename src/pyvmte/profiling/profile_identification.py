"""Setup for profiling identification function."""
from pyvmte.config import DGP_MST, IV_PAPER, SETUP_FIG5
from pyvmte.estimation.estimation import _compute_u_partition, _generate_basis_funcs

u_partition = _compute_u_partition(target=SETUP_FIG5.target, pscore_z=IV_PAPER.pscores)
bfuncs = _generate_basis_funcs("constant", u_partition=u_partition)

target = SETUP_FIG5.target
identified_estimands = SETUP_FIG5.identified_estimands
basis_funcs = bfuncs
m0_dgp = DGP_MST.m0
m1_dgp = DGP_MST.m1
instrument = IV_PAPER
