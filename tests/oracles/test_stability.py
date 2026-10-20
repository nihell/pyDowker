"""Stability: perturbing the matrix by at most eps moves every simplex, and every barcode, by at most eps.

f(sigma) = min_w max_{v in sigma} M[v, w] is 1-Lipschitz in the sup norm, and so is the m-th
smallest value used for m > 1. Hence |f_M(sigma) - f_M'(sigma)| <= ||M - M'||_inf for every
simplex, and by the stability theorem for two filtrations of one complex (Cohen-Steiner,
Edelsbrunner, Harer) the bottleneck distance in each dimension obeys the same bound.

The per-simplex check is exact in floating point if the bound is max|M' - M| computed in
floating point, not the drawn perturbation: filtration values are entries of M and M', and
rounding is monotone, so fl|f - f'| <= fl max|M' - M|. (Bounding by max|E| instead fails by
one ulp on tie-heavy inputs.) The bottleneck check allows 1e-12 for GUDHI's own arithmetic.
"""

import zlib

import gudhi
import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from tests.oracles import corpus

D = 2
CASES = corpus.real_matrices()
WITH_M = [(name, M, m) for name, M in CASES for m in (1, 2) if m <= M.shape[1]]


def perturb(M, seed):
    eps = 0.05 * max(float(np.ptp(M)), 1.0)
    M2 = M + np.random.default_rng(seed).uniform(-eps, eps, size=M.shape)
    return M2, float(np.max(np.abs(M2 - M)))


def tree(M, m=1):
    return DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel", m=m)


@pytest.mark.parametrize("name, M, m", WITH_M, ids=[f"{n}-m{m}" for n, _, m in WITH_M])
def test_filtration_values_move_at_most_eps(name, M, m):
    M2, bound = perturb(M, seed=zlib.crc32(f"{name}-{m}".encode()))
    a, b = tree(M, m), tree(M2, m)  # keep the trees alive while iterating (see CLAUDE.md)
    before = {tuple(s): f for s, f in a.get_simplices()}
    after = {tuple(s): f for s, f in b.get_simplices()}
    assert before.keys() == after.keys()
    assert max(abs(before[s] - after[s]) for s in before) <= bound


@pytest.mark.parametrize("name, M", CASES, ids=corpus.ids(CASES))
def test_bottleneck_distance_at_most_eps(name, M):
    M2, bound = perturb(M, seed=zlib.crc32(name.encode()))
    a, b = tree(M), tree(M2)
    a.compute_persistence(persistence_dim_max=True)
    b.compute_persistence(persistence_dim_max=True)
    for d in range(D + 1):
        dist = gudhi.bottleneck_distance(a.persistence_intervals_in_dimension(d),
                                         b.persistence_intervals_in_dimension(d))
        assert dist <= bound + 1e-12, f"H_{d}: {dist} > {bound}"
