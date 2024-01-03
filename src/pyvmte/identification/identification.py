"""Function for identification."""

from pyvmte.utilities import gamma_star


def identification(
    target_estimands,
    identified_estimands,
    basis_funcs,
    basis,
    m0_dgp,
    m1_dgp,
    shape_constraint=None,
    k0_size=None,
    k1_size=None,
    u_partition=None,
    u_lo_late_target=None,
    u_hi_late_target=None,
    u_lo_late_identified=None,
    u_hi_late_identified=None,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
    dz_cross=None,
    analytical_integration=False,
):
    help_vars = _get_helper_variables()

    values_identified = _compute_identified_estimands(
        identified_estimands,
        m0_dgp,
        m1_dgp,
        u_lo_late_identified,
        u_hi_late_identified,
        u_partition,
        support_z,
        pscore_z,
        pdf_z,
    )

    lp_inputs = _compute_lp_inputs()

    lp_results = _solve_lp()

    pass


def _get_helper_variables():
    pass


def _compute_identified_estimands(
    identified_estimands,
    m0_dgp,
    m1_dgp,
    u_lo,
    u_hi,
    u_part,
    support_z,
    pscore_z,
    pdf_z,
):
    """Wrapper for computing identified estimands based on provided dgp."""
    out = []
    for estimand in identified_estimands:
        result = _compute_estimand(
            estimand, m0_dgp, m1_dgp, u_lo, u_hi, u_part, support_z, pscore_z, pdf_z
        )
        out.append(result)

    return out


def _compute_estimand(
    estimand,
    m0,
    m1,
    u_lo=None,
    u_hi=None,
    u_part=None,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
):
    """Compute single identified estimand."""
    if estimand == "late":
        out = _compute_estimand_late(m0, m1, u_lo, u_hi, u_part)
    elif estimand == "iv_slope":
        out = _compute_estimand_iv_slope(m0, m1, u_part, support_z, pscore_z, pdf_z)
    elif estimand == "ols_slope":
        out = _compute_estimand_ols_slope(m0, m1, u_part, support_z, pscore_z, pdf_z)
    elif estimand == "cross":
        out = _compute_estimand_crossmoment(m0, m1)

    return out


def _compute_estimand_late(m0, m1, u_lo, u_hi, u_part):
    """Compute identified estimand for late."""
    a = gamma_star(m0, 0, estimand="late", u_lo=u_lo, u_hi=u_hi, u_part=u_part)
    b = gamma_star(m1, 1, estimand="late", u_lo=u_lo, u_hi=u_hi, u_part=u_part)

    return a + b


def _compute_estimand_iv_slope(m0, m1, u_part, support_z, pscore_z, pdf_z):
    """Compute identified estimand for iv slope."""
    a = gamma_star(
        m0,
        0,
        "iv_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    b = gamma_star(
        m1,
        1,
        "iv_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    return a + b


def _compute_estimand_ols_slope(m0, m1, u_part, support_z, pscore_z, pdf_z):
    """Compute identified estimand for ols slope."""
    a = gamma_star(
        m0,
        0,
        "ols_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    b = gamma_star(
        m1,
        1,
        "ols_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    return a + b


def _compute_estimand_crossmoment():
    """Compute identified estimand for crossmoment."""
    pass


def _compute_lp_inputs():
    """Compute inputs for the linear program."""
    pass


def _solve_lp():
    """Solve the linear program."""
    pass
