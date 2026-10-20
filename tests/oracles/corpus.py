"""Deterministic small inputs shared by the oracle and contract tests.

Covers degenerate shapes (one row, one witness), both rectangular orientations, tie-heavy
integer matrices, and attention-like matrices (-A for causal softmax rows with a sink on key 0).
"""

import numpy as np

SEED = 20261020
SHAPES = [(1, 1), (1, 4), (4, 1), (2, 2), (3, 5), (5, 3), (4, 6), (6, 4), (7, 7), (8, 3)]


def attention_like(rng, n, sink=2.0):
    """-A for a causal attention matrix A whose rows are softmax weights over keys 0..q.

    Key 0 gets a logit boost (an attention sink). Masked entries have A = 0, i.e. value 0,
    the top of the filtration.
    """
    logits = rng.normal(size=(n, n))
    logits[:, 0] += sink
    logits = np.where(np.tril(np.ones((n, n), dtype=bool)), logits, -np.inf)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    return -A + 0.0  # + 0.0 turns -0.0 into 0.0


def real_matrices():
    """[(name, float64 matrix)]: uniform, integer ties, attention-like."""
    rng = np.random.default_rng(SEED)
    cases = []
    for shape in SHAPES:
        cases.append((f"uniform{shape[0]}x{shape[1]}", rng.random(shape)))
        cases.append((f"ties{shape[0]}x{shape[1]}", rng.integers(0, 4, size=shape).astype(float)))
    for n in (4, 6, 7):
        cases.append((f"attention{n}", attention_like(rng, n)))
    return cases


def relations():
    """[(name, boolean matrix)]: random relations plus hand-made edge cases."""
    rng = np.random.default_rng(SEED + 1)
    cases = [(f"bernoulli{s[0]}x{s[1]}", rng.random(s) < 0.5) for s in SHAPES]
    cases += [
        ("disjoint_witnesses", np.eye(3, dtype=bool)),
        ("witnessless_row", np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)),
        ("complete", np.ones((4, 3), dtype=bool)),
    ]
    return cases


def ids(cases):
    return [name for name, *_ in cases]
