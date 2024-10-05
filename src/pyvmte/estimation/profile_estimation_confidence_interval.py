"""Profile Estimation Confidence Interval."""


from pyvmte.classes import Estimand
from pyvmte.config import RNG
from pyvmte.utilities import simulate_data_from_simple_model_dgp

dgp_params = {
    "y1_at": 1,
    "y0_at": 0,
    "y1_c": 0.5,
    "y0_c": 0,
    "y1_nt": 1,  # ensures we are at the upper bound of the identified set
    "y0_nt": 0,
}

data = simulate_data_from_simple_model_dgp(
    sample_size=10_000,
    rng=RNG,
    dgp_params=dgp_params,
)
kwargs = {
    "target": Estimand("late", u_hi_extra=0.2),
    "identified_estimands": [
        Estimand(
            "late",
        ),
    ],
    "basis_func_type": "constant",
    "y_data": data["y"],
    "z_data": data["z"],
    "d_data": data["d"],
    "confidence_interval": "bootstrap",
}
