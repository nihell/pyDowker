# pyDowker

Python package for filtered and bifiltered Dowker complexes of arbitrary relations given as
matrices (GUDHI simplex trees, RIVET bifiltrations, Euler characteristic profiles). Reference:
Hellmer and Spaliński, *Density Sensitive Bifiltered Dowker Complexes via Total Weight*,
arXiv:2405.15592. Main consumer: AttentionDowker, which builds Dowker complexes from transformer
attention matrices; its needs enter as requirements with `origin: attention/...`.

## Commands

```bash
pip install -e ".[test]"                        # once, in the main checkout
pytest                                          # oracles, contracts (per status), tooling
pytest tests/oracles                            # invariants only
python scripts/check_requirements.py            # validate requirements.yaml, show status and dependencies
python scripts/check_requirements.py --guard    # implementer branches: no protected path changed
```

pytest imports pyDowker from the checkout it runs in (`pythonpath = ["src"]` in pyproject), so
the same commands work in a worktree. Never `pip install` from a worktree: it would repoint the
shared environment there.

## Layout

- `src/pyDowker/DowkerComplex.py`: `DowkerComplex(rel_matrix)` with `create_simplex_tree`
  (`"Sublevel"`, `"TotalWeight"`, `"None"`), `create_rivet_bifiltration`,
  `euler_profile_contributions`, and `create_rivet_bifiltration_cpp`, which shells out to
  `./a.out` built from `MNeighborComplex.c++`.
- `src/pyDowker/TwoParameterUtils.py`: Hilbert function and Euler profile evaluation on grids.
- `examples/`: notebooks and the protein-ligand featurizer. Not tested. Several pass
  `max_filtration` to the constructor, which currently ignores it (DWK-004).
- `requirements.yaml`: what is being built and why. `docs/specs/`, `docs/decisions/`,
  `docs/reports/`: specs, decision records, experiment reports (each folder has a template).

## Mathematical conventions

- Rows are vertices, columns are witnesses: (v, w) is in R_t iff `M[v, w] <= t`. A simplex σ is
  in D(R_t) iff one witness relates to every vertex of σ.
- Sublevel value of σ: min over w of max over v in σ of `M[v, w]`. With `m`, the m-th smallest
  of those values, so σ needs m common witnesses.
- Superlevel data is negated before it arrives: attention enters as −A, stronger attention earlier.
- `max_dimension = D` builds the D-skeleton. Only H_d for d < D is the homology of the complex.
- GUDHI: always `compute_persistence(persistence_dim_max=True)`. The default skips the top
  dimension of the complex, which for a one-vertex complex is H_0. This silently broke Dowker
  duality checks on one-witness inputs.
- GUDHI: hold a simplex tree in a variable while iterating it. `get_simplices()` does not keep
  its tree alive, so `make_tree().get_simplices()` reads freed memory and can segfault
  (reproduced with GUDHI 3.13).
- Bifiltrations reverse the second axis: grade (a, b) means present at (x, m) iff a <= x and b >= m.
- Filtration values are matrix entries, so tests compare them exactly. Allow a tolerance only
  where arithmetic happens, and say in the test why.

## Invariants

Each is an oracle in `tests/oracles/` and must always pass: agreement with a brute-force
evaluation of the definitions (including cutoffs and the RIVET presence region), Dowker duality
of row and column sides in dimensions below the skeleton, Euler-Poincaré at every threshold,
and stability (filtration values and bottleneck distance move by at most ‖M − M'‖∞).

## Workflow

- **Registry.** Every change to `src/` traces to an entry in `requirements.yaml`. Its status
  drives its contract tests: not implemented means strict xfail, so the registry cannot drift
  from the code. When a hypothesis of a consumer is dropped, its `serves` entries show which
  requirements to drop with it.
- **Spec sessions** (Niklas with a frontier model at high effort) write specs, contracts, oracles
  and decisions, edit the registry, and interpret reports. Open decisions go to Niklas as a
  numbered list instead of being settled in passing.
- **implementer** subagent (Sonnet, own worktree): one accepted requirement at a time. A guard
  hook (`.claude/hooks/guard_protected_paths.py`) blocks edits to `tests/` except `tests/unit/`,
  `.claude/`, `docs/specs/`, `docs/decisions/` and `scripts/check_requirements.py`, allows only
  status flips in the registry, and will not let the subagent finish with protected changes.
  Claude Code runs a project subagent's hooks only after the folder is trusted.
- **reviewer** subagent (Opus, read-only) before merging. Hand it the requirement ID and the
  branch, not the implementer's explanation.
- Worktrees branch from the default branch: merge spec and contract changes to `main` before
  handing a requirement to the implementer.
- **Experiments:** copy `docs/reports/TEMPLATE.md`, write the prediction before running, record
  raw numbers, interpret afterwards. Measure memory in a fresh process per run (BENCH-001).

## Closed experiments

Record rejected approaches here with the evidence, so they are not tried again. None yet.
