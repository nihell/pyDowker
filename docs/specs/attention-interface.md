# What AttentionDowker needs from pyDowker

Status: draft
Requirements: DWK-001 (implemented), PKG-001 (accepted), DWK-006 (proposed), DWK-007 (proposed)
Date: 2026-10-20

## Problem

AttentionDowker tests whether Dowker filtrations of transformer attention carry signal for
hallucination detection (hypotheses H1 to H6), on GPT-2 small or Pythia-160M with HaluEval,
CPU only. It uses pyDowker's sublevel path once per head and needs a head-agreement
bifiltration. This spec fixes the interface so that Phase 0 can be written against it.

## Definitions

Attention of head h: A_h with rows (queries q) softmax-normalised over keys 0..q, and
A_h[q, k] = 0 for k > q (causal mask).

AttentionDowker passes M_h = −A_h: queries are rows (vertices), keys are columns (witnesses).
(q, k) is related at threshold t in [−1, 0] iff A_h[q, k] >= −t. A set σ of queries enters at

    f_h(σ) = min_k max_{q in σ} M_h[q, k] = −max_k min_{q in σ} A_h[q, k],

the moment some key receives at least that much attention from every query in σ.

**Side.** By functorial Dowker duality, `DowkerComplex(M_h)` and `DowkerComplex(M_h.T)` (keys as
vertices) have the same barcodes in dimensions below the skeleton. pyDowker enumerates subsets
of rows, so put the shorter side in the rows. This matters for rectangular prompt-response
blocks (H5). Compare sides only with `compute_persistence(persistence_dim_max=True)`.

**Mask.** Masked pairs have M = 0, the end of the filtration. While key 0 is a witness, every
query is related to it at t = 0, so encoding masked pairs as 0 or as +inf gives the same
filtration on [−1, 0].

**Sink.** Key 0 relates to every query from t_sink = max_q −A_h[q, 0] on. From there the complex
is the full simplex on all queries: no homology below the skeleton dimension. With a strong sink,
t_sink comes early and most other attention weights enter after it, so the signal has to come
from the short range below t_sink. See open decision 1 and the cost measurement in
`docs/reports/2026-10-20-baseline-cost.md`.

## Head-agreement bifiltration (DWK-006)

For k = 1..H let G_k[q, w] be the k-th smallest of M_1[q, w], ..., M_H[q, w], i.e. minus the
k-th largest attention weight on (q, w) across heads. Then

    R_{t,k} = {(q, w) : #{h : M_h[q, w] <= t} >= k} = {(q, w) : G_k[q, w] <= t},

and σ is in D(R_{t,k}) iff t >= t_k(σ) := min_w max_{q in σ} G_k[q, w].

G_1 <= G_2 <= ... <= G_H entrywise, so t_1(σ) <= ... <= t_H(σ): σ has grades
{(t_k(σ), k)} with the k-axis reversed, the same shape as the m-axis of
`create_rivet_bifiltration`. Slice k is the existing sublevel path applied to G_k, so the oracle
suite applies slice by slice and a contract can compare against
`reference.sublevel_filtration(G_k)`.

Agreement here is per pair: different queries of σ may be related to w through different sets
of k heads. See open decision 2.

## Interface

- Input per head: a float array of any floating dtype, shape (rows, columns) (DWK-001, done).
  For DWK-006: an array of shape (H, rows, columns).
- pyDowker imports and runs its GUDHI paths without pyrivet (PKG-001).
- DWK-006 output: open decision 3.

## Properties

For a DWK-006 contract once decisions 2 and 3 are made:
- slice k equals the brute-force sublevel filtration of G_k;
- t_k(σ) is nondecreasing in k for every σ;
- H = 1 reproduces the single-head tree;
- permuting heads changes nothing.

## Budget (DWK-007)

One layer (12 heads, n ≈ 512, max_dimension <= 2) in under 30 s on Niklas's laptop, measured
in a fresh process per run (BENCH-001). Baseline in the report above.

## Open decisions

1. **Sink handling** (AttentionDowker side): keep key 0; drop key 0 as a witness; drop it and
   renormalise rows; or change the weight transform (−log A, per-row ranks). Recommendation:
   run Phase 0 with "keep" and "drop key 0" and compare t_sink with where the H_1 bars lie.
2. **Pair-level or simplex-level agreement.** Simplex-level: σ is present at (t, k) iff one key w
   relates to all of σ in the same k heads. That is also a Dowker complex, with witnesses
   (key, k-subset of heads), so C(H, k) times as many witnesses (924 for H = 12, k = 6), and it
   is a subcomplex of the pair-level one. Recommendation: pair-level, because it reuses the
   sublevel path per slice and keeps the witness set small.
3. **DWK-006 output:** (a) one GUDHI tree per k; (b) a multipers multi-critical simplex tree;
   (c) the existing RIVET format. Recommendation: (a) as the tested output, plus a thin converter
   to multipers in AttentionDowker, after checking the current multipers API.
4. **Cutoff in Phase 0:** whether to pass `max_filtration`, and how to choose it (global
   quantile, below t_sink, per head). A cutoff changes which bars are visible.

## Out of scope

Running the models, tokenisation, HaluEval loading, vectorisation of diagrams, classifiers.
