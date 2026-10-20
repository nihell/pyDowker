"""DWK-003: euler_profile_contributions, evaluated with ecp_value, is the Euler characteristic profile.

At (x, m), with the second axis reversed (larger m is later), the profile is the Euler
characteristic of the full complex of simplices that have at least m witnesses within
threshold x: sum over sigma of (-1)^dim(sigma) [s_m(sigma) <= x], where s_m(sigma) is the
m-th smallest of max_{v in sigma} M[v, w].
"""

import numpy as np
import pytest

from pyDowker.DowkerComplex import DowkerComplex
from pyDowker.TwoParameterUtils import ecp_value, grid_ECP
from tests.oracles import corpus, reference

pytestmark = pytest.mark.req("DWK-003")


def test_profile_matches_brute_force_euler_characteristic():
    for name, M in corpus.real_matrices():
        n = M.shape[0]
        if n > 6:
            continue
        m_max = min(3, M.shape[1])
        contributions = DowkerComplex(M).euler_profile_contributions(m_max=m_max)
        xs = reference.thresholds(M.ravel())
        ms = list(range(1, m_max + 1))
        for x in xs:
            for m in ms:
                expected = sum((-1) ** (len(s) - 1) for s in reference.simplices(n, n - 1)
                               if reference.present(M, s, x, m))
                assert ecp_value(contributions, x, m) == expected, (name, x, m)
        grid = grid_ECP(contributions, np.array(xs), np.array(ms))
        assert grid.shape == (len(ms), len(xs))
        assert all(grid[j, i] == ecp_value(contributions, x, m)
                   for i, x in enumerate(xs) for j, m in enumerate(ms)), name
