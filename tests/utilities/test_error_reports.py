import numpy as np
import pytest
from pyvmte.classes import Estimand, Instrument
from pyvmte.estimation import estimation
from pyvmte.identification import identification
from pyvmte.utilities import (
    _error_report_basis_funcs,
    _error_report_confidence_interval,
    _error_report_estimand,
    _error_report_estimation_data,
    _error_report_instrument,
    _error_report_invalid_basis_func_type,
    _error_report_method,
    _error_report_mte_monotone,
    _error_report_mtr_function,
    _error_report_shape_constraints,
    _error_report_tolerance,
    _error_report_u_partition,
)


@pytest.mark.parametrize(
    "estimand",
    [
        "not an estimand",
        Estimand(esttype="hello_world"),
        Estimand(esttype="cross", dz_cross="hello_world"),  # type: ignore
        Estimand(esttype="late", u_lo=0.5, u_hi="daslk"),  # type: ignore
        Estimand(esttype="late", u_lo=0.5, u_hi=0.2),
        Estimand(esttype="late", u_lo=1.2, u_hi=1.4),
    ],
)
def test_error_report_estimand(estimand):
    error_report = _error_report_estimand(estimand, mode="estimation")
    print(error_report)  # noqa: T201
    assert error_report != ""


def test_error_report_invalid_basis_func_type():
    assert _error_report_invalid_basis_func_type("not a basis func type") != ""
    assert _error_report_invalid_basis_func_type("constant") == ""


@pytest.mark.parametrize(
    ("y_data", "z_data", "d_data"),
    [
        # Create some test cases
        # Case 1: One of the data is not a numpy array
        (1, [1, 2, 3], [0, 1, 0]),
        ([1, 2, 3], 1, [0, 1, 0]),
        ([1, 2, 3], [1, 2, 3], 1),
        # Case 2: Arrays, but different lengths
        (np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([0, 1, 0])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 1, 0, 1])),
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3]), np.array([0, 1, 0])),
        # Case 3: Arrays, same length, but d has values other than 0 and 1
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 1, 2])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 1, 0.5])),
    ],
)
def test_error_report_estimation_data(y_data, z_data, d_data):
    error_report = _error_report_estimation_data(y_data, z_data, d_data)
    print(error_report)  # noqa: T201
    assert error_report != ""


def test_error_report_method():
    assert _error_report_method("not a method") != ""
    assert _error_report_method(123142) != ""
    assert _error_report_method("copt") == ""
    assert _error_report_method("copt") == ""


def test_error_report_tolerance():
    assert _error_report_tolerance("not a tolerance") != ""
    assert _error_report_tolerance(0) != ""
    assert _error_report_tolerance(-1) != ""
    assert _error_report_tolerance(0.5) == ""


@pytest.mark.parametrize(
    "u_part",
    [
        "not a u partition",
        np.array([0.5, 0.5]),
        np.array([0.5, 0.2]),
        np.array([1.2, 1.4]),
    ],
)
def test_error_report_u_partition(u_part):
    error_report = _error_report_u_partition(u_part)
    print(error_report)  # noqa: T201
    assert error_report != ""


@pytest.mark.parametrize(
    "valid_u_part",
    [
        # Create some test cases for valid u_partitions
        np.array([0, 0.35, 0.65, 0.7, 1]),
        np.array([0, 0.5, 1]),
        np.array([0, 1]),
    ],
)
def test_error_report_u_partition_valid(valid_u_part):
    error_report = _error_report_u_partition(valid_u_part)
    print(error_report)  # noqa: T201
    assert error_report == ""


# Generate some test inputs that make estimation() fail
# First generate a valid input as a fixtures
# Then modify the valid input to make estimation() fail
# Then call estimation() and assert that it fails
# Finally, check that the error_report is not empty


@pytest.fixture()
def valid_input():
    return {
        "target": Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
        "identified_estimands": [Estimand(esttype="iv_slope")],
        "basis_func_type": "constant",
        "y_data": np.array([1, 2, 3]),
        "z_data": np.array([1, 2, 3]),
        "d_data": np.array([0, 1, 0]),
        "tolerance": 0.1,
        "u_partition": np.array([0, 0.35, 0.65, 0.7, 1]),
        "method": "highs",
        "confidence_interval": None,
    }


@pytest.mark.parametrize(
    "invalid_input",
    [
        # Create some test cases
        # Case 1: y_data is not a numpy array
        {"y_data": 1},
        # Case 2: z_data is not a numpy array
        {"z_data": 1},
        # Case 3: d_data is not a numpy array
        {"d_data": 1},
        # Case 4: tolerance is not a number
        {"tolerance": "not a number"},
        # Case 5: tolerance is not positive
        {"tolerance": 0},
        # Case 6: u_partition is not a numpy array
        {"u_partition": "not a u partition"},
        # Case 7: u_partition is not valid
        {"u_partition": np.array([0.5, 0.5])},
        {"u_partition": np.array([0.5, 0.2])},
        {"u_partition": np.array([1.2, 1.4])},
        {"target": "not an estimand"},
        {"target": Estimand(esttype="hello_world")},
        {"target": Estimand(esttype="cross", dz_cross="hello_world")},  # type: ignore
        {"target": Estimand(esttype="late", u_lo=0.5)},
        {"target": Estimand(esttype="late", u_hi=0.5)},
        {"target": Estimand(esttype="late", u_lo=0.5, u_hi="daslk")},  # type: ignore
        {"target": Estimand(esttype="late", u_lo=0.5, u_hi=0.2)},
        {"target": Estimand(esttype="late", u_lo=1.2, u_hi=1.4)},
        {"identified_estimands": "not an estimand"},
        {"identified_estimands": Estimand(esttype="hello_world")},
        {"identified_estimands": Estimand(esttype="cross", dz_cross="hello_world")},  # type: ignore
        {"identified_estimands": Estimand(esttype="late", u_lo=0.5)},
        {"identified_estimands": Estimand(esttype="late", u_hi=0.5)},
        {"identified_estimands": Estimand(esttype="late", u_lo=0.5, u_hi="daslk")},  # type: ignore
        {"identified_estimands": Estimand(esttype="late", u_lo=0.5, u_hi=0.2)},
        {"identified_estimands": Estimand(esttype="late", u_lo=1.2, u_hi=1.4)},
        ({"confidence_interval": "not valid"}),
        ({"confidence_interval": True}),
        ({"confidence_interval": 1}),
    ],
)
def test_estimation_invalid_input(valid_input, invalid_input):
    valid_input.update(invalid_input)
    with pytest.raises(ValueError):  # noqa: PT011
        estimation(**valid_input)


@pytest.mark.parametrize(
    ("invalid_1", "invalid_2", "invalid_3"),
    [
        # Create some test cases
        # Case 1: y_data is not a numpy array
        ({"y_data": 1}, {"z_data": 1}, {"d_data": 1}),
        # Case 2: z_data is not a numpy array
        ({"z_data": 1}, {"d_data": 1}, {"tolerance": "not a number"}),
        # Case 3: d_data is not a numpy array
        (
            {"d_data": 1},
            {"tolerance": "not a number"},
            {"u_partition": "not a u partition"},
        ),
        # Case 4: tolerance is not a number
        (
            {"tolerance": "not a number"},
            {"u_partition": "not a u partition"},
            {"target": "not an estimand"},
        ),
        # Case 5: tolerance is not positive
        (
            {"tolerance": 0},
            {"target": "not an estimand"},
            {"identified_estimands": "not an estimand"},
        ),
        # Case 6: u_partition is not a numpy array
        (
            {"u_partition": "not a u partition"},
            {"identified_estimands": "not an estimand"},
            {"basis_func_type": "not a basis func type"},
        ),
        # Case 7: u_partition is not valid
        (
            {"u_partition": np.array([0.5, 0.5])},
            {"basis_func_type": "not a basis func type"},
            {"method": "not a method"},
        ),
        (
            {"u_partition": np.array([0.5, 0.2])},
            {"method": "not a method"},
            {"y_data": 1},
        ),
        ({"u_partition": np.array([1.2, 1.4])}, {"y_data": 1}, {"z_data": 1}),
        ({"target": "not an estimand"}, {"z_data": 1}, {"d_data": 1}),
        (
            {"target": Estimand(esttype="hello_world")},
            {"d_data": 1},
            {"tolerance": "not a number"},
        ),
        (
            {"target": Estimand(esttype="cross", dz_cross="hello_world")},
            {"tolerance": "not a number"},
            {"u_partition": "not a u partition"},
        ),  # type: ignore
    ],
)
def test_estimation_multiple_invalid_input(
    valid_input,
    invalid_1,
    invalid_2,
    invalid_3,
):
    valid_input.update(invalid_1)
    valid_input.update(invalid_2)
    valid_input.update(invalid_3)
    with pytest.raises(ValueError):  # noqa: PT011
        estimation(**valid_input)


@pytest.mark.parametrize(
    "basis_funcs",
    [
        # Create some test cases
        "not a list",
        [],
        [{"type": "constant", "u_lo": 0.0, "u_hi": 2.0}],
        [
            {"type": "dasds", "u_lo": 0.0, "u_hi": 0.5},
            {"type": "constant", "u_lo": 0.5, "u_hi": 1.0},
        ],
        [
            {"type": "constant", "u_lo": 0.0, "u_hi": 0.5},
            {"type": "constant", "u_lo": -1, "u_hi": 0.7},
            {"type": "constant", "u_lo": 0.7, "u_hi": 1.0},
        ],
    ],
)
def test_error_report_basis_funcs(basis_funcs):
    error_report = _error_report_basis_funcs(basis_funcs)
    print(error_report)  # noqa: T201
    assert error_report != ""


@pytest.mark.parametrize(
    "mtr_function",
    [
        "not a function",
        1,
        lambda x, y: x + y,
    ],
)
def test_error_report_mtr_function(mtr_function):
    error_report = _error_report_mtr_function(mtr_function)
    print(error_report)  # noqa: T201
    assert error_report != ""


@pytest.mark.parametrize(
    "instrument",
    [
        "not an instrument",
        1,
        Instrument(
            support=np.array([1, 2, 3]),
            pmf=np.array([-0.1, 0.3, 0.2]),
            pscores=np.array([0.5, 0.3, 0.2]),
        ),
        Instrument(
            support=np.array([1, 2, 3]),
            pmf=np.array([0.1, 0.3, 0.2]),
            pscores=np.array([0.5, 0.3, 0.7]),
        ),
    ],
)
def test_error_report_instrument(instrument):
    error_report = _error_report_instrument(instrument)
    print(error_report)  # noqa: T201
    assert error_report != ""


@pytest.mark.parametrize(
    "shape_constraints",
    [
        "not a tuple",
        (1, 2),
        ("not a shape constraint", "not a shape constraint"),
        ("increasing", "not a shape constraint"),
        ("not a shape constraint", "increasing"),
    ],
)
def test_error_report_shape_constraints(shape_constraints):
    error_report = _error_report_shape_constraints(shape_constraints)
    assert error_report != ""


# Add tests that check that identification returns error for invalid inputs
@pytest.fixture()
def valid_input_identification():
    return {
        "target": Estimand(esttype="late", u_lo=0.35, u_hi=0.9),
        "identified_estimands": [Estimand(esttype="iv_slope")],
        "basis_funcs": [{"type": "constant", "u_lo": 0.0, "u_hi": 1.0}],
        "m0_dgp": lambda x: x,
        "m1_dgp": lambda x: x,
        "instrument": Instrument(
            support=np.array([1, 2, 3]),
            pmf=np.array([0.1, 0.3, 0.6]),
            pscores=np.array([0.5, 0.3, 0.2]),
        ),
        "u_partition": np.array([0, 0.35, 0.65, 0.7, 1]),
    }


@pytest.mark.parametrize(
    ("invalid_input1", "invalid_input2"),
    [
        ({"target": "not an estimand"}, {"identified_estimands": "not an estimand"}),
        (
            {"target": Estimand(esttype="hello_world")},
            {"identified_estimands": "not an estimand"},
        ),
        (
            {"target": Estimand(esttype="cross", dz_cross="hello_world")},
            {"identified_estimands": "not an estimand"},
        ),  # type: ignore
        (
            {"target": Estimand(esttype="late", u_lo=0.5)},
            {"identified_estimands": "not an estimand"},
        ),
        (
            {"target": Estimand(esttype="late", u_hi=0.5)},
            {"identified_estimands": "not an estimand"},
        ),
        (
            {"target": Estimand(esttype="late", u_lo=0.5, u_hi="daslk")},
            {"identified_estimands": "not an estimand"},
        ),  # type: ignore
        (
            {"target": Estimand(esttype="late", u_lo=0.5, u_hi=0.2)},
            {"identified_estimands": "not an estimand"},
        ),
        (
            {"target": Estimand(esttype="late", u_lo=1.2, u_hi=1.4)},
            {"identified_estimands": "not an estimand"},
        ),
        (
            {"target": "not an estimand"},
            {"identified_estimands": Estimand(esttype="hello_world")},
        ),
        (
            {"target": Estimand(esttype="hello_world")},
            {"identified_estimands": Estimand(esttype="hello_world")},
        ),
        (
            {"target": Estimand(esttype="cross", dz_cross="hello_world")},
            {"identified_estimands": Estimand(esttype="hello_world")},
        ),  # type: ignore
        # Invalid instrment
        ({"instrument": "not an instrument"}, {"u_partition": "not a u partition"}),
        ({"instrument": 1}, {"u_partition": "not a u partition"}),
        # Invalid mte_monotone
        ({"mte_monotone": "not valid"}, None),
        ({"mte_monotone": 1}, None),
        ({"mte_monotone": False}, None),
    ],
)
def test_identification_invalid_input(
    valid_input_identification,
    invalid_input1,
    invalid_input2,
):
    valid_input_identification.update(invalid_input1)
    if invalid_input2 is not None:
        valid_input_identification.update(invalid_input2)
    with pytest.raises(ValueError):  # noqa: PT011
        identification(**valid_input_identification)


def test_error_report_mte_monotone():
    assert _error_report_mte_monotone("not a boolean") != ""
    assert _error_report_mte_monotone(1) != ""
    assert _error_report_mte_monotone(mte_monotone=True) != ""
    assert _error_report_mte_monotone(mte_monotone=False) != ""

    for value in [None, "increasing", "decreasing"]:
        assert _error_report_mte_monotone(value) == ""


def test_error_report_confidence_interval():
    invalid_args = ["numerical_delta", "analytical_delta", 1, True]

    for arg in invalid_args:
        assert _error_report_confidence_interval(arg) != ""

    valid_args = [None, "bootstrap", "subsampling", "rescaled_bootstrap"]

    for arg in valid_args:
        assert _error_report_confidence_interval(arg) == ""
