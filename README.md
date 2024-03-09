# pyvmte

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/buddejul/pyvmte/main.svg)](https://results.pre-commit.ci/latest/github/buddejul/pyvmte/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/buddejul/pyvmte/graph/badge.svg?token=T6D31ZBXK9)](https://codecov.io/gh/buddejul/pyvmte)

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate pyvmte
```

To build the project, type

```console
$ pytask
```

To reduce runtime it is recommended to use the
[pytask-parallel](https://github.com/pytask-dev/pytask-parallel) plug-in:

```console
$ pytask -n <workers>
```

where `workers` is the number of workers.

With parallelization the project builds in 5-10 minutes on my machine using 11 workers.

To reduce run-time it is also possible to adjust the simulation settings in
`config_mc_by_size.py` and `config_mc_by_target.py`:

```python
MC_SAMPLE_SIZES = [500, 2500, 10000]

MONTE_CARLO_BY_SIZE = MonteCarloSetup(
    sample_size=10_000,
    repetitions=10_000,
)

MONTE_CARLO_BY_TARGET = MonteCarloSetup(
    sample_size=10_000,
    repetitions=1_000,
    u_hi_range=np.arange(0.35, 1, 0.05),
)
```

Reducing the `repetitions` always works. Only be careful with really small sample sizes
which can result in errors because estimators are undefined (the linear programs do not
have a solution).

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
