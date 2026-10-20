"""Measurement script for docs/reports/2026-10-20-baseline-cost.md.

    python docs/reports/2026-10-20-baseline-cost.py            # every run, one fresh process each
    python docs/reports/2026-10-20-baseline-cost.py --verify   # prototypes vs pyDowker, small inputs
    python docs/reports/2026-10-20-baseline-cost.py --child BUILDER N KEEP SINK

Each run builds the sublevel simplex tree (max_dimension = 2, m = 1) of a synthetic
attention-like matrix (-A for causal softmax rows with a sink on key 0) and computes H0 and H1.
KEEP is the fraction of unmasked (query, key) pairs related at the cutoff (1.0: no cutoff).
SINK is `keep`, or `drop` to remove key 0 as a witness. Peak RSS is read inside the child,
which is fresh, so it is that run's own peak.

BUILDER is `pydowker` (DowkerComplex.create_simplex_tree) or one of three prototypes that
produce the same tree: `vectorized` (one numpy step for all cofaces of a node), `batched`
(plus one insert_batch call for all top-dimensional cofaces of a node), `restricted` (plus,
under a cutoff, only the witnesses that still relate to the node). Prototypes are not part
of the package.
"""

import json
import resource
import subprocess
import sys
import time

import gudhi
import numpy as np

D = 2
RUNS = (
    [("pydowker", n, 1.0, "keep") for n in (32, 64, 128, 192, 256)]
    + [("pydowker", n, 0.05, "keep") for n in (128, 256, 512)]
    + [("pydowker", n, 0.05, "drop") for n in (256, 512)]
    + [(b, 256, 1.0, "keep") for b in ("vectorized", "batched")]
    + [(b, 512, 0.05, s) for b in ("vectorized", "batched", "restricted") for s in ("keep", "drop")]
)


def attention_like(n, seed=0, sink=2.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n))
    logits[:, 0] += sink
    mask = np.tril(np.ones((n, n), dtype=bool))
    logits = np.where(mask, logits, -np.inf)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    return -A + 0.0, mask


def prototype(M, cutoff, batch=False, restrict=False):
    st, n = gudhi.SimplexTree(), M.shape[0]

    def node(sigma, wv):
        st.insert(sigma, wv.min())
        start = sigma[-1] + 1
        if len(sigma) > D or start >= n:
            return
        W = np.flatnonzero(wv <= cutoff) if restrict else slice(None)
        cofaces = np.maximum(M[start:][:, W], wv[W])
        values = cofaces.min(axis=1)
        keep = np.flatnonzero(values <= cutoff)
        if batch and len(sigma) == D:
            if len(keep):
                vertices = np.empty((D + 1, len(keep)), dtype=np.int32)
                vertices[:D] = np.array(sigma, dtype=np.int32)[:, None]
                vertices[D] = start + keep
                st.insert_batch(vertices, values[keep])
            return
        for j in keep:
            if restrict:
                child = np.full_like(wv, np.inf)
                child[W] = cofaces[j]
            else:
                child = cofaces[j]
            node(sigma + [start + int(j)], child)

    for k in range(n - 1, -1, -1):
        if M[k].min() <= cutoff:
            node([k], M[k])
    return st


def build(builder, M, cutoff):
    if builder == "pydowker":
        from pyDowker.DowkerComplex import DowkerComplex
        return DowkerComplex(M).create_simplex_tree(max_dimension=D, filtration="Sublevel",
                                                    max_filtration=cutoff)
    return prototype(M, cutoff, batch=builder in ("batched", "restricted"),
                     restrict=builder == "restricted")


def matrix(n, keep, sink):
    M, mask = attention_like(n)
    if sink == "drop":
        M, mask = M[:, 1:].copy(), mask[:, 1:]
    cutoff = np.inf if keep >= 1.0 else float(np.quantile(M[mask], keep))
    return M, cutoff


def child(builder, n, keep, sink):
    import pyDowker.DowkerComplex  # noqa: F401  import cost stays out of the timing
    M, cutoff = matrix(n, keep, sink)
    t0 = time.perf_counter()
    st = build(builder, M, cutoff)
    t1 = time.perf_counter()
    st.compute_persistence(persistence_dim_max=False)  # H0 and H1 of the 2-skeleton
    t2 = time.perf_counter()
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
    print(json.dumps({"builder": builder, "n": n, "keep": keep, "sink": sink,
                      "simplices": st.num_simplices(), "build_s": round(t1 - t0, 2),
                      "persistence_s": round(t2 - t1, 2), "peak_rss_mib": round(peak_mib)}))


def verify():
    for n, keep, sink in [(64, 1.0, "keep"), (64, 0.3, "keep"), (96, 0.05, "drop"), (128, 0.1, "keep")]:
        M, cutoff = matrix(n, keep, sink)
        st = build("pydowker", M, cutoff)  # iterate only trees held in a variable
        ref = {tuple(s): f for s, f in st.get_simplices()}
        for b in ("vectorized", "batched", "restricted"):
            st = build(b, M, cutoff)
            same = ref == {tuple(s): f for s, f in st.get_simplices()}
            print(f"n={n} keep={keep} sink={sink} {b}: {'identical' if same else 'DIFFERENT'}")
            assert same


def main():
    for builder, n, keep, sink in RUNS:
        out = subprocess.run([sys.executable, __file__, "--child", builder, str(n), str(keep), sink],
                             capture_output=True, text=True, check=True)
        print(out.stdout.strip(), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
    elif sys.argv[1:2] == ["--verify"]:
        verify()
    else:
        main()
