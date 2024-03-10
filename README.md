# pyvmte

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/buddejul/pyvmte/main.svg)](https://results.pre-commit.ci/latest/github/buddejul/pyvmte/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/buddejul/pyvmte/graph/badge.svg?token=T6D31ZBXK9)](https://codecov.io/gh/buddejul/pyvmte)

## Project

This project implements a method for inference about general treatment effects in
instrumental variables settings first proposed in
[Mogstad, Santos, Torgovitsky 2018 Econometrica](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA15463)
(henceforth MST). The main novel results reported here are Monte Carlo simulations
corresponding to the data generating process of the paper. In particular, I report
extensive simulations corresponding to the *identification* result reported in Figure 5
of the paper.

The goal of MST is to make inference on parameters that are generally not
point-identified in instrumental variable settings. For example, a well-known result is
that we can only point-identify the local average treatment effect (LATE) for a given
complier subpopulation in a binary-instrument binary-treatment setting. However, the
researcher might be interested in the LATE for a different subpopulation or the average
treatment effect (ATE).

The general idea of the paper is to *set*-identify a target parameter, where the
identified set is constrained by the estimators we can point-identify. Intuitively, we
need to make some assumptions about unobservables to provide a set for the target
parameter. However, all parameters that we can point-identify put restrictions on these
unobservables. The main contribution of MST is to show that all identified and target
parameters can be written as linear maps of so-called *marginal treatment response (MTR)
functions* in a binary choice model. Hence, a combination of data moments and
assumptions on MTR functions imply an identified set for the target parameter. For a
more detailed introduction to the method see the report in this project.

## Implementation

All sets in MST (identified or estimated) are implicitly defined by linear programs
(LPs). Thus, the key programming task is to compute the inputs into the linear program.
I then pass these into `scipy.optimize.linprog` which is the scipy wrapper for several
LP solvers, including `highs` which I use as the standard. I also implement the `copt`
solver, which generally performs best for a range of problems (e.g. see these
[benchmarks](https://mattmilten.github.io/mittelmann-plots/)). However, for the small
size of the problems in my simulations I did not see any performance differences (if
anything, scipy has the faster API).

Following MST, I split the code into a section `identification` and `estimation`. The
former implements pure identification results for a known DGP, while the latter
implements estimation of the identified set based on data. Both are based on LPs similar
in spirit but with slightly different constraints. In particular, `estimation` has to
deal with sampling uncertainty since in any finite sample the constraints will only be
satisfied approximately. For details see the report.

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
which can result in errors because estimators might be undefined (the linear programs do
not have a solution).

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
