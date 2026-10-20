"""DWK-002: filtration="None" builds the Dowker complex of the relation, honouring m.

The complex contains a simplex iff its vertices have at least m common witnesses;
float input is first turned into the relation M <= level.
"""

import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

pytestmark = pytest.mark.req("DWK-002")
D = 3


def simplex_set(st):
    return {tuple(s) for s, _ in st.get_simplices()}


def test_boolean_relations_give_the_dowker_complex_with_at_least_m_witnesses():
    for name, R in corpus.relations():
        for m in (1, 2):
            st = DowkerComplex(R).create_simplex_tree(max_dimension=D, filtration="None", m=m)
            assert simplex_set(st) == reference.dowker_complex(R, D, m), f"{name}, m={m}"


def test_float_input_is_thresholded_at_level():
    for name, M in corpus.real_matrices():
        level = float(np.median(M))
        st = DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="None", level=level)
        assert simplex_set(st) == reference.dowker_complex(M <= level, D), name
