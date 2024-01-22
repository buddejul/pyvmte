import numpy as np
import pandas as pd  # type: ignore

from pyvmte.utilities import load_paper_dgp


def simulate_data_from_paper_dgp(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    dgp = load_paper_dgp()

    z_dict = dict(zip(dgp["support_z"], dgp["pscore_z"]))

    sampled = np.random.choice(dgp["support_z"], size=sample_size, p=dgp["pdf_z"])

    # FIXME 60% here
    # TODO probably fastest: just sample from tuples....
    pscores_corresponding = np.array([z_dict[i] for i in sampled])

    data["z"] = sampled
    data["pscore_z"] = pscores_corresponding

    data["u"] = rng.uniform(size=sample_size)

    data["d"] = data["u"] < data["pscore_z"]

    m0 = dgp["m0"]
    m1 = dgp["m1"]

    # FIXME 20% here
    data["y"] = np.where(data["d"] == 0, m0(data["u"]), m1(data["u"]))

    data["pscore_z"] = data["pscore_z"].astype(float)
    data["z"] = data["z"].astype(int)
    data["u"] = data["u"].astype(float)
    data["d"] = data["d"].astype(int)
    data["y"] = data["y"].astype(float)

    return data


def fast_simulate(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    support = np.array([0, 1, 2])
    pmf = np.array([0.5, 0.4, 0.1])
    pscores = np.array([0.35, 0.6, 0.7])

    choices = np.hstack([support.reshape(-1, 1), pscores.reshape(-1, 1)])

    # Draw random ndices
    idx = np.random.choice(len(support), size=sample_size, p=pmf)

    data = choices[idx]

    # Put data into df
    data = pd.DataFrame(data, columns=["z", "pscore_z"])

    data["u"] = rng.uniform(size=sample_size)

    data["d"] = data["u"] < data["pscore_z"]

    dgp = load_paper_dgp()

    m0 = dgp["m0"]
    m1 = dgp["m1"]

    # FIXME 20% here
    data["y"] = np.where(data["d"] == 0, m0(data["u"]), m1(data["u"]))

    data["pscore_z"] = data["pscore_z"].astype(float)
    data["z"] = data["z"].astype(int)
    data["u"] = data["u"].astype(float)
    data["d"] = data["d"].astype(int)
    data["y"] = data["y"].astype(float)

    return data


def vectorized_fast_simulate(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    support = np.array([0, 1, 2])
    pmf = np.array([0.5, 0.4, 0.1])
    pscores = np.array([0.35, 0.6, 0.7])

    choices = np.hstack([support.reshape(-1, 1), pscores.reshape(-1, 1)])

    # Draw random ndices
    idx = np.random.choice(len(support), size=sample_size, p=pmf)

    data = choices[idx]

    # Put data into df
    data = pd.DataFrame(data, columns=["z", "pscore_z"])

    data["u"] = rng.uniform(size=sample_size)

    data["d"] = data["u"] < data["pscore_z"]

    dgp = load_paper_dgp()

    m0 = dgp["m0"]
    m1 = dgp["m1"]

    vectorized_m0 = np.vectorize(m0)
    vectorized_m1 = np.vectorize(m1)

    # FIXME 20% here
    data["y"] = np.where(
        data["d"] == 0, vectorized_m0(data["u"]), vectorized_m1(data["u"])
    )

    data["pscore_z"] = data["pscore_z"].astype(float)
    data["z"] = data["z"].astype(int)
    data["u"] = data["u"].astype(float)
    data["d"] = data["d"].astype(int)
    data["y"] = data["y"].astype(float)

    return data
