import numpy as np
import pytest


@pytest.fixture
def preprint_fig5_relation():
    """4×4 boolean relation from Figure 5 of the preprint (used for duality examples)."""
    return np.array(
        [[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=bool
    )


@pytest.fixture
def bool_relation():
    """5×4 boolean relation used across multiple tests."""
    return np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )


@pytest.fixture
def float_relation():
    """5×4 float relation used across multiple tests."""
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
