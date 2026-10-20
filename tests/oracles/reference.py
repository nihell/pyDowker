"""Brute-force reference constructions for the oracle and contract tests.

Every function enumerates vertex subsets with itertools and evaluates the defining formula
in plain Python. Slow on purpose: use on inputs with at most about 8 rows.

Conventions (pyDowker's): rows of the matrix are vertices, columns are witnesses, and
(v, w) is in the relation at threshold t iff M[v][w] <= t. A simplex is a sorted tuple of
row indices; max_dimension = D means simplices with at most D + 1 vertices.
"""

from itertools import combinations
from math import inf, isfinite


def as_lists(M, kind=float):
    return [[kind(x) for x in row] for row in M]


def simplices(n_rows, max_dimension):
    for k in range(1, max_dimension + 2):
        yield from combinations(range(n_rows), k)


def common_values(M, sigma):
    """For each witness w, the threshold max_{v in sigma} M[v][w] at which w witnesses all of sigma."""
    M = M if isinstance(M, list) else as_lists(M)
    return [max(M[v][w] for v in sigma) for w in range(len(M[0]))]


def sublevel_value(M, sigma, m=1):
    """Entry threshold of sigma: the m-th smallest common value (m = 1: min over witnesses)."""
    values = sorted(common_values(M, sigma))
    return values[m - 1] if m <= len(values) else inf


def sublevel_filtration(M, max_dimension, m=1):
    """{sigma: entry threshold} for every simplex that enters at a finite threshold."""
    M = as_lists(M)
    out = {}
    for sigma in simplices(len(M), max_dimension):
        t = sublevel_value(M, sigma, m)
        if isfinite(t):
            out[sigma] = t
    return out


def present(M, sigma, x, m):
    """sigma is in the complex at (x, m): at least m witnesses relate to all of sigma at threshold x."""
    return sum(c <= x for c in common_values(M, sigma)) >= m


def witness_count(R, sigma):
    return sum(all(R[v][w] for v in sigma) for w in range(len(R[0])))


def dowker_complex(R, max_dimension, m=1):
    """Unfiltered Dowker complex of a boolean relation: simplices with at least m common witnesses."""
    R = as_lists(R, bool)
    return {s for s in simplices(len(R), max_dimension) if witness_count(R, s) >= m}


def total_weight_filtration(R, max_dimension):
    """{sigma: -(number of common witnesses)} over the simplices of the Dowker complex."""
    R = as_lists(R, bool)
    out = {}
    for sigma in simplices(len(R), max_dimension):
        k = witness_count(R, sigma)
        if k >= 1:
            out[sigma] = float(-k)
    return out


def euler_characteristic(filtration, t):
    """Euler characteristic of the subcomplex {sigma : filtration[sigma] <= t}."""
    return sum((-1) ** (len(s) - 1) for s, f in filtration.items() if f <= t)


def betti_alternating_sum(intervals_by_dim, t):
    """sum_d (-1)^d beta_d(t), with beta_d(t) = #{bars [b, e) in dimension d : b <= t < e}."""
    return sum((-1) ** d * sum(1 for b, e in bars if b <= t < e)
               for d, bars in intervals_by_dim.items())


def thresholds(values):
    """Every distinct value, the midpoints between them, and one point below and above."""
    v = sorted(set(float(x) for x in values))
    if not v:
        return []
    mids = [(a + b) / 2 for a, b in zip(v, v[1:])]
    return [v[0] - 1.0] + sorted(v + mids) + [v[-1] + 1.0]
