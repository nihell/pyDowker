"""Euler-Poincare at every threshold: counting simplices and reading the barcode agree.

chi(K_t) = sum_sigma (-1)^dim(sigma) [f(sigma) <= t] must equal sum_d (-1)^d beta_d(t), where
beta_d(t) counts the bars [b, e) in dimension d with b <= t < e. This checks that the tree is
a valid filtration and that barcodes are read with the right conventions: half-open bars,
essential bars, and the top dimension. For a D-skeleton the identity holds for the skeleton
itself, which needs its H_D, hence persistence_dim_max=True (the horizon effect described in
test_dowker_duality.py).
"""

import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus, reference

D = 3
CASES = [(name, M, m) for name, M in corpus.real_matrices() for m in (1, 2) if m <= M.shape[1]]


@pytest.mark.parametrize("name, M, m", CASES, ids=[f"{n}-m{m}" for n, _, m in CASES])
def test_euler_characteristic_from_simplices_equals_alternating_betti_sum(name, M, m):
    st = DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel", m=m)
    st.compute_persistence(persistence_dim_max=True)
    filtration = {tuple(s): f for s, f in st.get_simplices()}
    bars = {d: st.persistence_intervals_in_dimension(d) for d in range(D + 1)}
    for t in reference.thresholds(filtration.values()):
        assert reference.euler_characteristic(filtration, t) == reference.betti_alternating_sum(bars, t), t
