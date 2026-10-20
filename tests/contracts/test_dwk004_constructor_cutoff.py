"""DWK-004: max_filtration passed to the constructor prunes the sublevel tree.

Covers only the constructor-only case. Precedence when the method also receives a cutoff,
and cutoffs for the RIVET paths, are open questions in the requirement's notes.
"""

import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

pytestmark = pytest.mark.req("DWK-004")
D = 2


def test_constructor_cutoff_keeps_exactly_the_simplices_at_or_below_it():
    for name, M in corpus.real_matrices():
        cutoff = float(np.median(M))
        st = DowkerComplex(M, max_filtration=cutoff).create_simplex_tree(max_dimension=D,
                                                                         filtration="Sublevel")
        got = {tuple(s): f for s, f in st.get_simplices()}
        expected = {s: t for s, t in reference.sublevel_filtration(M, D).items() if t <= cutoff}
        assert got == expected, name
