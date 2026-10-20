"""DWK-005: the TotalWeight filtration lives on the Dowker complex.

Every simplex with at least one common witness appears at -(number of common witnesses);
simplices without a common witness, including vertices whose row is empty, do not appear.
"""

import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

pytestmark = pytest.mark.req("DWK-005")
D = 3


def test_total_weight_filtration_matches_reference():
    for name, R in corpus.relations():
        st = DowkerComplex(R).create_simplex_tree(max_dimension=D, filtration="TotalWeight")
        got = {tuple(s): f for s, f in st.get_simplices()}
        assert got == reference.total_weight_filtration(R, D), name
