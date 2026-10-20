# pyDowker

[![PyPI version](https://img.shields.io/pypi/v/pyDowker.svg)](https://pypi.org/project/pyDowker/)
[![Tests](https://github.com/nihell/pyDowker/actions/workflows/test.yml/badge.svg)](https://github.com/nihell/pyDowker/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/pydowker/badge/?version=latest)](https://pydowker.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2405.15592-b31b1b.svg)](https://arxiv.org/abs/2405.15592)

**pyDowker** computes density-sensitive bifiltered Dowker complexes for topological data analysis (TDA).

Given a relation matrix $\Lambda \in \mathbb{R}^{n \times m}$ (e.g. distances from $n$ data points to $m$ witnesses), pyDowker builds:

- A **sublevel-filtered** Dowker complex (one-parameter persistence, compatible with [gudhi](https://gudhi.inria.fr))
- A **total-weight-filtered** Dowker complex (density-sensitive, one-parameter)
- A **bifiltration** combining both (two-parameter persistence via [multipers](https://github.com/DavidLapous/multipers))

The C++ backend (pybind11) makes the bifiltration computation fast enough for real datasets.

This package accompanies the paper:

> Niklas Hellmer & Jan Spaliński,
> *Density Sensitive Bifiltered Dowker Complexes via Total Weight*,
> [arXiv:2405.15592](https://arxiv.org/abs/2405.15592)

## Installation

```bash
pip install pyDowker
```

With optional multipers support for two-parameter persistence:

```bash
pip install "pyDowker[multipers]"
```

Building from source requires a C++17 compiler and CMake ≥ 3.18.

## Quick start

### One-parameter persistence (sublevel filtration)

```python
import numpy as np
import gudhi as gd
from pyDowker import DowkerComplex

# rel[i, j] = distance from data point i to witness j
rel = np.array([[1, 2, 3, 4],
                [2, 3, 4, 5],
                [2, 0, 2, 6],
                [4, 7, 1, 1],
                [3, 1, 4, 1]], dtype=np.float64)

dc = DowkerComplex(rel)
st = dc.create_simplex_tree(filtration="Sublevel", m=1, max_dimension=2)
st.compute_persistence()
gd.plot_persistence_diagram(st.persistence())
```

### Two-parameter bifiltration

```python
# Returns dict mapping simplex → list of (r, m) appearance bidegrees
bifilt = dc.create_rivet_bifiltration_cpp(max_dimension=2, m_max=5)

# Or directly as a multipers SimplexTreeMulti (requires pip install pyDowker[multipers])
st_multi = dc.create_multipers_bifiltration(max_dimension=2, m_max=5)
```

### Filtration modes

| Mode | Input | Description |
|---|---|---|
| `"Sublevel"` | float matrix | Simplex appears at the smallest $r$ where $m$ witnesses are within $r$ |
| `"TotalWeight"` | bool or float+level | Simplex appears at $-($ number of common witnesses $)$ |
| `"None"` | bool or float+level | Unfiltered complex: simplex present iff it has $\geq m$ witnesses |

## Documentation

Full documentation, including a theory primer and API reference, is at
[pydowker.readthedocs.io](https://pydowker.readthedocs.io).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to build from source, run tests, and build the docs.

## Citation

If you use pyDowker in your research, please cite:

```bibtex
@misc{hellmer2024density,
  title   = {Density Sensitive Bifiltered {D}owker Complexes via Total Weight},
  author  = {Niklas Hellmer and Jan Spali{\'n}ski},
  year    = {2024},
  eprint  = {2405.15592},
  archivePrefix = {arXiv},
  primaryClass  = {math.AT},
  url     = {https://arxiv.org/abs/2405.15592},
}
```
