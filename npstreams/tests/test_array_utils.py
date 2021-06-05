import numpy as np
from npstreams import nan_to_num


def test_nan_to_num_generic():
    """Test that NaNs are replaced with a fill value"""
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = nan_to_num(np.array([0]) / 0.0, fill_value=14)
    assert vals[0] == 14


def test_nan_to_num_integer():
    """Test that nan_to_num on integers does nothing"""
    vals = nan_to_num(1)
    assert vals == 1
    vals = nan_to_num([1])
    assert np.allclose(vals, np.array([1]))


def test_nan_to_num_complex_good():
    """Test nan_to_num on complex input"""
    vals = nan_to_num(1 + 1j)
    assert vals == 1 + 1j
