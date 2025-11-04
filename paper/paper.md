---
title: 'X: tensor‑rank cross and matrix‑train optimizers for sample‑efficient black‑box search'
tags:
  - python
  - optimization
  - tensor networks
authors:
  - name: Roman Ellerbrock
    orcid: 0000-0003-3555-6263
    affiliation: 1
  - name: Alexander C. Paul
    orcid: 0000-0002-7547-445X
    affiliation: 1
  - name: Aleksandr Berezutskii
    orcid: 0009-0002-8149-135X
    affiliation: 1
affiliations:
 - name: Terra Quantum AG, Freddie-Mercury Str. 5, Munich, DE 80979
   index: 1
date: 01 November 2025
bibliography: paper.bib
---

# Summary

**X** is a lightweight Python library that implements two optimizers for black-box optimization on discrete grids: **Tensor‑Rank Cross (TRC)** and **Matrix‑Train (MT)**. Both methods build low‑rank tensor representations of the objective from a small number of function evaluations, then extract candidate optima directly from the representations. The package targets use‑cases where objective gradients are unavailable or unreliable, and where each function call is expensive (simulation, hardware loop, or batched evaluation).

# Statement of need

Optimization practitioners routinely face high‑dimensional, nonconvex objectives with tight evaluation budgets. General‑purpose gradient‑free methods (e.g., direct search or evolutionary strategies) are robust but often require many evaluations to locate good solutions. Low‑rank tensor decompositions offer an orthogonal strategy: approximate the discretized objective on a grid with a compact representation and use its structure to propose maximizers/minimizers efficiently. Despite promising results in recent literature [oseledets2010tt][sozykin2022ttopt][dolgov2025tensor], ...

Here’s a version that matches what the package actually does today:

# Functionality

* **Tensor-Rank Cross (TRC) optimizer.** Performs rank-(r) cross updates on a user-supplied product grid. Each sweep alternates **one-leg mutations** (cross-sampling one primitive while holding others fixed) with **selection** via max-volume or assignment on the evaluated candidate matrix. It returns the updated rank-(r) skeleton (pivot set) and the current incumbent. Rank, number of sweeps, grid resolution, and seeds are user-controlled.

* **Matrix-Train (MT) optimizer.** Optimizes an (N)-site matrix-train directly on the grid. The core step **recombines** left/right blocks to form an (r\times r) candidate matrix and uses the **Hungarian (linear-sum) assignment** to pick the next rank-(r) slice, interleaved with one-leg mutation updates. Grouped/segment assignments are supported via a simple grouped assignment routine (experimental).

* **Search spaces.** Any space that can be discretized into **per-dimension 1D grids**: continuous boxes (e.g., uniform linspaces), categorical/discrete sets, or mixed discrete–continuous spaces by mixing primitives. Custom per-dimension grids are accepted.

* **Budgets & reproducibility.** Work is bounded by rank, number of sweeps, and grid size; the framework tracks objective-call counts and supports **deterministic seeding** for fair comparisons (common-random-numbers across methods).

* **Benchmarks & baselines.** The repo includes a benchmarking suite with CSV outputs and plots, comparing TRC/MT against **TTOpt** and **SciPy** baselines (Differential Evolution and Dual Annealing).


# Minimal example

```python
import numpy as np

```
Following the minimal example above, users can easily adapt X to ... Additional usage examples as well as the detailed API documentation are provided in the
[X Documentation](https://mdopt.readthedocs.io/en/latest/).

# References
