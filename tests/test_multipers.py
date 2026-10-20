"""Tests for multipers integration. Skipped if multipers is not installed."""

import numpy as np
import pytest

mp = pytest.importorskip("multipers", reason="multipers not installed")

from pyDowker import DowkerComplex


@pytest.fixture
def float_relation():
    return np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [2, 0, 2, 6],
            [4, 7, 1, 1],
            [3, 1, 4, 1],
        ],
        dtype=np.float64,
    )


def test_returns_simplex_tree_multi(float_relation):
    st = DowkerComplex(float_relation).create_multipers_bifiltration(max_dimension=2)
    # Cython type is template-specialized; check duck-type interface instead
    assert hasattr(st, "num_parameters") and hasattr(st, "is_kcritical")


def test_num_parameters(float_relation):
    st = DowkerComplex(float_relation).create_multipers_bifiltration(max_dimension=2)
    assert st.num_parameters == 2


def test_is_kcritical(float_relation):
    st = DowkerComplex(float_relation).create_multipers_bifiltration(max_dimension=2)
    assert st.is_kcritical


def test_simplex_2_4_first_appearance(float_relation):
    """[2,4]: m=1 appearance should be at r=1.0 (first axis), m=-1 (second axis)."""
    st = DowkerComplex(float_relation).create_multipers_bifiltration(max_dimension=2)
    appearances = st.filtration([2, 4])
    first_r_values = [a[0] for a in appearances]
    assert pytest.approx(1.0) in first_r_values


def test_second_axis_is_negative_m(float_relation):
    """The second filtration parameter is stored as -m (negative witness count)."""
    st = DowkerComplex(float_relation).create_multipers_bifiltration(
        max_dimension=1, m_max=3
    )
    for simplex, _ in [([0], None), ([1], None)]:
        for app in st.filtration(simplex):
            assert app[1] <= 0, "Second parameter should be non-positive (-m)"


def test_raises_without_multipers(monkeypatch):
    """ImportError raised when multipers is not available."""
    import sys
    import importlib

    mod = sys.modules["pyDowker.DowkerComplex"]
    monkeypatch.setattr(mod, "_HAS_MULTIPERS", False)
    rel = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ImportError, match="multipers"):
        DowkerComplex(rel).create_multipers_bifiltration(max_dimension=1)
