# Notebooks

Run from the repository root:

```bash
uv run --group docs jupyter lab notebooks
```

| folder | what's in it | open when |
| --- | --- | --- |
| `tutorials/` | one stage of the library per notebook, in order | you want to learn the API |
| `examples/` | published problems solved end to end | you want to see it work once |

## Tutorials

Start with [`01_overview`](tutorials/01_overview.ipynb), then read in order — each stage's output is
the next stage's input.

| | stage | what you learn |
| --- | --- | --- |
| [`01_overview`](tutorials/01_overview.ipynb) | — | the pipeline as a figure, and all five stages in one cell |
| [`02_ode`](tutorials/02_ode.ipynb) | 1 | `from_sympy` vs `from_strings`, multi-index grids, and what the input validation rejects |
| [`03_linearization`](tutorials/03_linearization.ipynb) | 2 | Koopman, Liouville and KvN as three orderings of one template |
| [`04_basis`](tutorials/04_basis/index.ipynb) | 3 | the shared interface, then [Carleman](tutorials/04_basis/carleman.ipynb), [PSC](tutorials/04_basis/psc_carleman.ipynb), [Gauss–Hermite](tutorials/04_basis/gaussian_hermite.ipynb) |
| [`05_matrix`](tutorials/05_matrix.ipynb) | 4 | elements and blocks on a grid of $10^{9}$ sites |
| [`06_vector`](tutorials/06_vector.ipynb) | 5 | initial values and observables as coefficients |
| [`07_boundary_conditions`](tutorials/07_boundary_conditions.ipynb) | — | the declarative `conditions` API: Dirichlet and state-dependent constraints on a periodic grid |
| [`08_end_to_end`](tutorials/08_end_to_end.ipynb) | — | the full pipeline twice: from a known point, then from a density |
| [`09_serialization`](tutorials/09_serialization.ipynb) | — | `dump_json` and `from_json` on every object the pipeline takes in or hands back, and what a string leaves out |

## Examples

<details open>
<summary><b>Basic examples</b></summary>

| | problem | what it adds |
| --- | --- | --- |
| [`sho_kinetic_energy`](examples/sho_kinetic_energy.ipynb) | 1-D harmonic oscillator | all three linearizations, each against a closed form |
| [`kvn_heisenberg`](examples/kvn_heisenberg.ipynb) | 3-D harmonic oscillator | KvN evolution; energy conserved exactly |
| [`psc_logistic_switching`](examples/psc_logistic_switching.ipynb) | scalar logistic equation | when moving the pivot helps, and when it does not |

</details>

<details>
<summary><b>Advanced examples</b></summary>

| | problem | what it adds |
| --- | --- | --- |
| [`kpp_fisher_1d`](examples/kpp_fisher_1d.ipynb) | KPP–Fisher, Fig. 3 of [arXiv:2508.01640](https://arxiv.org/abs/2508.01640) | 88 element calls instead of all $9139^{2}$ matrix entries |
| [`kpp_fisher_2d`](examples/kpp_fisher_2d.ipynb) | the same equation, $8\times8$ grid | the same assembly, unchanged in 2-D |
| [`kpp_fisher_position_dependent_1d`](examples/kpp_fisher_position_dependent_1d.ipynb) | coefficients varying by site | shifting the grid no longer maps one entry onto another |
| [`kpp_fisher_position_dependent_2d`](examples/kpp_fisher_position_dependent_2d.ipynb) | a rectangular inclusion, $12\times12$ grid | a read-out along a line through the inclusion |
| [`psc_fig3_kpp_fisher`](examples/psc_fig3_kpp_fisher.ipynb) | KPP–Fisher on a ring, Fig. 3 of [arXiv:2411.16233](https://arxiv.org/abs/2411.16233) | Carleman at truncation order 3, against PSC at orders 3 and 5 |
| [`psc_fig4_phase_field`](examples/psc_fig4_phase_field.ipynb) | cubic phase-field model, Fig. 4 of the same paper | a cubic term raises degree by two, so truncation cuts two levels |

</details>
