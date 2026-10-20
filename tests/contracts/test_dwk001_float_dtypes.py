"""DWK-001: the sublevel path accepts any floating dtype and runs on NumPy >= 2.

One test on purpose: under strict xfail a contract must fail as a whole while the
requirement is unmet, so cases that already pass are not split into separate tests.
"""

import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

pytestmark = pytest.mark.req("DWK-001")
D = 2


def test_floating_dtypes_give_the_reference_filtration_and_booleans_are_rejected():
    for name, M in corpus.real_matrices():
        for dtype in (np.float16, np.float32, np.float64):
            X = M.astype(dtype)
            st = DowkerComplex(X).create_simplex_tree(max_dimension=D, filtration="Sublevel")
            got = {tuple(s): f for s, f in st.get_simplices()}
            assert got == reference.sublevel_filtration(X, D), f"{name} as {dtype.__name__}"

    with pytest.raises(TypeError):
        DowkerComplex(np.ones((3, 3), dtype=bool)).create_simplex_tree(max_dimension=1,
                                                                       filtration="Sublevel")
