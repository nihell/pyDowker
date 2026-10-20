# Mathematical background

This page gives a brief overview of the mathematics behind pyDowker.
For full details and proofs, see [arXiv:2405.15592](https://arxiv.org/abs/2405.15592).

## Relations and Dowker complexes

Let $R \subseteq X \times Y$ be a relation between a finite set of *data points* $X$ and a finite set of *witnesses* $Y$.

The **Dowker complex** $\mathcal{D}(X, Y, R)$ is the simplicial complex on vertex set $X$ where a finite subset $\sigma \subseteq X$ is a simplex if and only if there exists a common witness $y \in Y$ such that $(x, y) \in R$ for every $x \in \sigma$.

**Dowker duality** (Dowker 1952) states that $\mathcal{D}(X, Y, R)$ and $\mathcal{D}(Y, X, R^T)$ are homotopy equivalent.

## Sublevel filtration

When $R$ is given by sublevels of a matrix $\Lambda \in \mathbb{R}^{X \times Y}$ (i.e. $(x,y) \in R_r \Leftrightarrow \Lambda_{xy} \leq r$), we obtain a **filtered** Dowker complex $\mathcal{D}(X, Y, R_\bullet)$.

The filtration value of a simplex $\sigma$ at witness multiplicity $m$ is

$$
r_m(\sigma) = \min_{S \subseteq Y,\, |S| = m} \max_{x \in \sigma,\, y \in S} \Lambda_{xy},
$$

i.e. the smallest $r$ at which $\sigma$ has $m$ common witnesses.

## Total-weight filtration

The **total weight** of a simplex $\sigma$ at threshold $r$ is the number of common witnesses:

$$
w_r(\sigma) = |\{y \in Y : (x, y) \in R_r \text{ for all } x \in \sigma\}|.
$$

Taking superlevels of $-w_r$ yields the *total-weight filtration*.

This filtration is an instance of **Sheehy's multicover filtration** (Theorem 3.3 in the paper).

## The bifiltration

Combining both axes gives the **measure Dowker bifiltration**

$$
\mathcal{MD}(X, \mu) = \{(r, m) \mapsto \mathcal{D}(X, Y, R_r) \text{ restricted to simplices with } \geq m \text{ witnesses}\}.
$$

A simplex $\sigma$ appears at bidegree $(r, m)$ if and only if it has at least $m$ common witnesses all within distance $r$.

**Theorem 4.4** (Stability): the change of the bifiltration under perturbations of the data is controlled by the Hausdorff distance and the Prokhorov metric.

## Algorithm

Algorithm 1 (Section 5) computes the appearances of all simplices in the bifiltration in a single recursive traversal.  The C++ backend in pyDowker implements this algorithm directly, giving a significant speedup over the pure-Python version for larger inputs.
