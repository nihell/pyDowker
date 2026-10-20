# Baseline cost of the Python sublevel path

Date: 2026-10-20 · Code: pyDowker 7f043c3 plus the DWK-001 fix from the following commit
(without it the sublevel path does not run on NumPy 2) · Requirements: DWK-007, DWK-008, ENG-001
Machine: Claude's cloud sandbox, x86_64, 2 cores, 7 GiB, Linux; not Niklas's laptop ·
Versions: Python 3.13.16, numpy 2.5.4, gudhi 3.13.0
Command: `python docs/reports/2026-10-20-baseline-cost.py` (timings) and `--verify` (prototypes)

## Question and prediction

How far is the current Python path from DWK-007's budget, one 12-head layer at n ≈ 512 and
max_dimension 2 in under 30 s?

Prediction before the first run: without a cutoff the work is one numpy step per simplex, so
the cost follows C(n,1) + C(n,2) + C(n,3) at roughly 5 to 10 µs per simplex, whatever the
attention values: 15 to 30 s per head at n = 256 and minutes per head at n = 512. The cutoff,
sink and prototype runs were added after the first results and had no prior prediction.

## Method

- Input: one synthetic matrix per n (seed 0). For each query q, logits N(0, 1) over keys 0..q
  with +2 on key 0 (an attention sink), softmax over the row, M = −A. Real heads look different.
- Cutoff: the KEEP-quantile of M over unmasked pairs, so KEEP = 0.05 relates the strongest 5% of
  pairs. "Sink dropped" removes key 0 as a witness before the quantile is taken.
- One run per row, each in a fresh process: build time, persistence time for H0 and H1
  (`persistence_dim_max=False`), peak RSS from `getrusage` inside that process. No repetitions,
  so run-to-run noise is not estimated; an earlier run of the same rows differed by up to 11% on
  rows longer than one second.
- Prototypes, defined in the script: *vectorized* computes all cofaces of a node in one numpy
  step; *batched* also inserts all top-dimensional cofaces of a node with one `insert_batch`
  call; *restricted* also uses, under a cutoff, only the witnesses that still relate to the
  node. `--verify` confirmed that all three give exactly pyDowker's tree (simplices and values)
  on four smaller inputs.

## Results

pyDowker, no cutoff, sink kept:

| n | simplices | build (s) | persistence (s) | peak RSS (MiB) |
|---:|---:|---:|---:|---:|
| 32 | 5,488 | 0.03 | 0.00 | 64 |
| 64 | 43,744 | 0.18 | 0.01 | 67 |
| 128 | 349,632 | 1.80 | 0.14 | 88 |
| 192 | 1,179,808 | 6.79 | 0.62 | 143 |
| 256 | 2,796,416 | 15.40 | 2.02 | 249 |

pyDowker, 5% cutoff:

| n | sink | simplices | build (s) | persistence (s) | peak RSS (MiB) |
|---:|---|---:|---:|---:|---:|
| 128 | kept | 88,396 | 0.99 | 0.05 | 70 |
| 256 | kept | 961,933 | 10.67 | 0.51 | 129 |
| 512 | kept | 7,350,209 | 97.33 | 6.01 | 551 |
| 256 | dropped | 99,627 | 5.95 | 0.04 | 72 |
| 512 | dropped | 1,277,978 | 69.56 | 1.08 | 152 |

Prototypes (build time; persistence and RSS as for pyDowker on the same tree):

| builder | n | cutoff | sink | build (s) | pyDowker build (s) | speed-up |
|---|---:|---|---|---:|---:|---:|
| vectorized | 256 | none | kept | 7.22 | 15.40 | 2.1× |
| batched | 256 | none | kept | 2.98 | 15.40 | 5.2× |
| vectorized | 512 | 5% | kept | 29.69 | 97.33 | 3.3× |
| batched | 512 | 5% | kept | 19.39 | 97.33 | 5.0× |
| restricted | 512 | 5% | kept | 6.80 | 97.33 | 14× |
| vectorized | 512 | 5% | dropped | 12.68 | 69.56 | 5.5× |
| batched | 512 | 5% | dropped | 11.73 | 69.56 | 5.9× |
| restricted | 512 | 5% | dropped | 2.92 | 69.56 | 24× |

## Interpretation

- The prediction held for the uncut path: 5.1 to 5.8 µs per simplex for n = 128 to 256.
  Extrapolated to n = 512 (22.4 million simplices): about 2.5 to 3 minutes per head including
  persistence, 30 to 35 minutes per 12-head layer, and about 1.5 GiB peak per head. That is
  roughly 65 times the budget.
- A cutoff helps less than the simplex counts suggest, for two reasons. With the sink kept, the
  strongest 5% of pairs at n = 512 still give 7.35 million simplices, a third of the full
  2-skeleton, because key 0 relates to most queries at that level and so witnesses most query
  triples. Dropping the sink leaves 1.28 million simplices, but build time falls only from 97 s
  to 70 s: the recursion calls numpy once per candidate coface before pruning, so its cost
  follows the candidates, not the simplices it keeps.
- The prototypes remove that overhead. At n = 512 with the sink dropped and a 5% cutoff, the
  restricted builder needs 2.9 s plus 1.1 s persistence per head, about 48 s for 12 heads on one
  core of this sandbox. Heads are independent, so with two heads at a time that is about 24 s,
  inside the budget. With the sink kept it is about 13 s per head.
- Not shown by these numbers: real attention, any cutoff rule other than a global 5% quantile,
  run-to-run variance, or Niklas's laptop.

## Consequences

- DWK-007 stays proposed under the pull policy, but meeting it in Python now looks plausible.
- DWK-008 (proposed) records the three prototype techniques; ENG-001 is now blocked by DWK-008.
- `docs/decisions/0002-native-engine-scope.md` uses these numbers.
- Sink handling (open decision 1 in `docs/specs/attention-interface.md`) is a cost question as
  well as a signal question: dropping the sink cut the simplex count 5.8× at n = 512 and the
  time per head 1.5× (pyDowker) to 3.3× (restricted prototype).
- Next run: GPT-2 small attention on HaluEval prompts, one layer, on Niklas's laptop, once
  BENCH-001 exists.
