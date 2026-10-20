"""The sublevel constructions agree with a brute-force evaluation of their definitions.

For a matrix M (rows = vertices, columns = witnesses) the Dowker sublevel filtration gives
a simplex sigma the threshold min_w max_{v in sigma} M[v, w]; with m > 1 it takes the m-th
smallest of the values max_{v in sigma} M[v, w] instead of the minimum. pyDowker computes
this by depth-first recursion with numpy; tests/oracles/reference.py evaluates the formula
directly. Filtration values are matrix entries, so the comparisons are exact.
"""

import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

D = 3
CASES = corpus.real_matrices()
WITH_M = [(name, M, m) for name, M in CASES for m in (1, 2) if m <= M.shape[1]]


def tree_values(st):
    return {tuple(s): f for s, f in st.get_simplices()}


@pytest.mark.parametrize("name, M, m", WITH_M, ids=[f"{n}-m{m}" for n, _, m in WITH_M])
def test_sublevel_tree_matches_reference(name, M, m):
    st = DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel", m=m)
    assert tree_values(st) == reference.sublevel_filtration(M, D, m)


@pytest.mark.parametrize("name, M", CASES, ids=corpus.ids(CASES))
def test_method_cutoff_keeps_exactly_the_simplices_at_or_below_it(name, M):
    cutoff = float(np.median(M))
    st = DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel",
                                              max_filtration=cutoff)
    expected = {s: t for s, t in reference.sublevel_filtration(M, D).items() if t <= cutoff}
    assert tree_values(st) == expected


@pytest.mark.parametrize("name, M", CASES, ids=corpus.ids(CASES))
def test_rivet_bifiltration_has_the_reference_presence_region(name, M):
    """Grades (a, b) are read with the second axis reversed: sigma is present at (x, m)
    iff some grade has a <= x and b >= m. Redundant grades are allowed."""
    m_max = min(3, M.shape[1])
    bf = DowkerComplex(M).create_rivet_bifiltration(max_dimension=D, m_max=m_max)
    grades = {tuple(s): a for s, a in zip(bf.simplices, bf.appearances)}
    assert set(grades) == set(reference.simplices(M.shape[0], D))
    for sigma, appearances in grades.items():
        for x in reference.thresholds(M.ravel()):
            for m in range(1, m_max + 1):
                got = any(a <= x and b >= m for a, b in appearances)
                assert got == reference.present(M, sigma, x, m), (sigma, x, m)
