"""Functorial Dowker duality: the row side and the column side have the same barcodes.

For R_t = {(v, w) : M[v, w] <= t}, Dowker's theorem gives D(R_t) ~ D(R_t^T), naturally in t
(Chowdhury-Memoli, functorial Dowker theorem). The persistence modules are isomorphic, so
the barcodes of DowkerComplex(M) and DowkerComplex(M.T) coincide.

Two GUDHI details decide whether the comparison means anything:
* max_dimension = D builds the D-skeleton, whose H_D is not the complex's H_D, so only
  dimensions d < D are compared.
* compute_persistence() skips the top dimension *of the complex* unless
  persistence_dim_max=True. An input with one witness has a one-vertex column side, and
  with the default its H_0 bar silently disappears.
Filtration values are matrix entries, so barcodes are compared exactly. GUDHI drops
zero-length bars by default, which matches what a module isomorphism preserves.
"""

import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus

D = 3
CASES = corpus.real_matrices()


def barcodes(M):
    st = DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel")
    st.compute_persistence(persistence_dim_max=True)
    return {d: sorted(map(tuple, st.persistence_intervals_in_dimension(d))) for d in range(D)}


@pytest.mark.parametrize("name, M", CASES, ids=corpus.ids(CASES))
def test_row_and_column_barcodes_agree_below_the_skeleton_dimension(name, M):
    rows, columns = barcodes(M), barcodes(M.T.copy())
    for d in range(D):
        assert rows[d] == columns[d], f"H_{d}"
