# pyDowker

**pyDowker** is a Python package (with a C++ backend) for constructing density-sensitive bifiltered Dowker complexes, as introduced in:

> Niklas Hellmer & Jan Spaliński,
> *Density Sensitive Bifiltered Dowker Complexes via Total Weight*,
> [arXiv:2405.15592](https://arxiv.org/abs/2405.15592)

## What is a Dowker complex?

Given a relation $R \subseteq X \times Y$ between a set of *data points* $X$ and a set of *witnesses* $Y$, the **Dowker complex** $\mathcal{D}(X, Y, R)$ is the simplicial complex on $X$ whose simplices are the subsets of $X$ that share a common witness in $Y$.

This package provides:

- A **sublevel filtration** of the Dowker complex by a weight function on $R$ (encodes proximity).
- A **total-weight filtration** by the number of shared witnesses (encodes density).
- A **bifiltration** combining both axes simultaneously, enabling two-parameter persistent homology via [multipers](https://github.com/DavidLapous/multipers) or [RIVET](https://rivet.readthedocs.io/).

## Installation

```bash
pip install pyDowker
```

With optional multipers support:

```bash
pip install "pyDowker[multipers]"
```

## Quick start

```python
import numpy as np
from pyDowker import DowkerComplex

# rel[i, j] = distance from data point i to witness j
rel = np.array([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=np.float64)

dc = DowkerComplex(rel)

# One-parameter persistence (sublevel filtration)
st = dc.create_simplex_tree(filtration="Sublevel", m=1, max_dimension=2)
st.compute_persistence()
print(st.persistence())

# Bifiltration (sublevel × witness-count)
bifilt = dc.create_rivet_bifiltration_cpp(max_dimension=2, m_max=3)
```

## Contents

```{toctree}
:maxdepth: 2

theory
api
examples
changelog
```
