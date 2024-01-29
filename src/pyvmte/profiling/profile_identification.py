"""Setup for profiling identification function."""
from pyvmte.config import IV_PAPER, SETUP_FIG5
from pyvmte.estimation.estimation import _compute_u_partition, _generate_basis_funcs
from pyvmte.utilities import load_paper_dgp

DGP = load_paper_dgp()

u_partition = _compute_u_partition(target=SETUP_FIG5.target, pscore_z=IV_PAPER.pscores)
bfuncs = _generate_basis_funcs("constant", u_partition=u_partition)

target = SETUP_FIG5.target
identified_estimands = SETUP_FIG5.identified_estimands
basis_funcs = bfuncs
m0_dgp = DGP["m0"]
m1_dgp = DGP["m1"]
instrument = IV_PAPER
