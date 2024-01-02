"""Utilities used in various parts of the project."""


def gamma_star(
    md,
    d,
    estimand,
    u_lo=0,
    u_hi=1,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
    dz_cross=None,
    analyt_int=False,
    u_part=None,
    u_part_lo=None,
    u_part_hi=None,
):
    """Compute gamma* for a given MTR function and estimand
    Args:
        md (function): MTR function
        d (np.int): value of the treatment
        estimand (str): the estimand to compute
        u_lo (float): lower bound of late target
        u_hi (float): upper bound of late target
        support_z (np.array): support of the instrument
        pscore_z (np.array): propensity given the instrument
        pdf_z (np.array): probability mass function of the instrument
        dz_cross (list): list of tuples of the form (d_spec, z_spec) for cross-moment
        analyt_int (Boolean): Whether to integrate manually or use analytic results
    """

    if estimand not in ["iv_slope", "late", "ols_slope", "cross"]:
        raise ValueError(
            "estimand must be either 'iv_slope', 'late', 'ols_slope' or 'cross'"
        )

    if estimand in ["iv_slope", "ols_slope"]:
        if support_z is None or pscore_z is None or pdf_z is None:
            raise ValueError(
                "support_z, pscore_z, and pdf_z must be specified for iv_slope or ols_slope"
            )

    if estimand == "cross":
        if dz_cross is None:
            raise ValueError("dz_cross must be specified for cross-moment")

    if analyt_int == True and u_part is None:
        raise ValueError("u_part must be specified for cs basis")

    if estimand == "late":
        return integrate.quad(lambda u: md(u) * s_late(d, u, u_lo, u_hi), u_lo, u_hi)[0]

    # Do integration manually via scipy integrate
    if analyt_int == False:
        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)

            if d == 0:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_iv_slope(z, ez, cov_dz)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] > u:
                        return md(u) * s_iv_slope(z, ez, cov_dz)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                # need to condition on z
                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_ols_slope(d, ed, var_d)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] > u:
                        return md(u) * s_ols_slope(d, ed, var_d)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

        if estimand == "cross":
            if d == 0:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_cross(d, z, dz_cross)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] >= u:
                        return md(u) * s_cross(d, z, dz_cross)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

    # Use analytic results on constant spline basis
    if analyt_int == True:
        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_iv_slope(z, ez, cov_dz)
                        * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_iv_slope(z, ez, cov_dz)
                        * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )

        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_ols_slope(d, ed, var_d)
                        * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_ols_slope(d, ed, var_d)
                        * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )

        if estimand == "cross":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j] * s_cross(d, z, dz_cross) * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j] * s_cross(d, z, dz_cross) * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )
