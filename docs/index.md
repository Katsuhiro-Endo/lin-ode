# lin-ode

Linearize a system of nonlinear ODEs and read any matrix element of the result in closed form,
without the grid size entering the cost.

$$\frac{d}{dt}\mathbf{x}(t)=\mathbf{F}\bigl(\mathbf{x}(t)\bigr),
\qquad \mathbf{x}(t)\in\mathbb{R}^{N},$$

with $\mathbf{F}$ polynomial in the state. A PDE becomes such a system once space is discretised, so
$N$ is the number of grid points.

Terms and symbols: [vocabulary and symbols](concepts.md).

## Install

Clone the repository, then install it from that checkout. To work in it:

```bash
uv sync --locked
```

To use it from another project, give the path to the checkout:

```bash
uv add /path/to/lin-ode          # or: pip install /path/to/lin-ode
```

## Quick start

Discretised KPP–Fisher, $\dot u_i = D(u_{i-1}-2u_i+u_{i+1})/h^{2}+Qu_i+Ru_i^{2}$, on four sites
with the coefficients left symbolic:

```python
import sympy as sp

from lin_ode import OrdinaryDifferentialSystem
from lin_ode.basis import CarlemanBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman

D, Q, R, h = sp.symbols("D Q R h")
u, i = sp.IndexedBase("u"), sp.Idx("i", (0, 3))
rhs = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2 + Q * u[i] + R * u[i] ** 2
ode = OrdinaryDifferentialSystem.from_sympy(rhs, u, (i,))

matrix = OperatorMatrix(CarlemanBasis(n_vars=4, degree=2), Koopman().create_operator(ode))
```

Two basis-state labels name an element, each a list of `(site, power)` pairs:

```python
matrix[(2, 2), (2, 1)]      # <u_2^2| L |u_2>  ->  R
matrix[(1, 1), (2, 1)]      # <u_1  | L |u_2>  ->  D/h**2
matrix[(2, 1), (2, 1)]      # <u_2  | L |u_2>  ->  -2*D/h**2 + Q
```

`OperatorMatrix` is lazy, and it derives each element from the two labels on demand.
Widen the grid to a billion sites and the same element still returns in a hundredth of a second:

```python
big_i = sp.Idx("i", (0, 10**9 - 1))
big = OrdinaryDifferentialSystem.from_sympy(rhs.subs(i, big_i), u, (big_i,))
OperatorMatrix(CarlemanBasis(10**9, 2), Koopman().create_operator(big))[(2, 2), (2, 1)]   # R
```

Change the basis or the linearization by changing one class name.

## Where to go next

| | |
| --- | --- |
| [Overview: the five stages](../notebooks/tutorials/01_overview.ipynb) | One ODE goes through the five stages that turn it into a matrix, with a notebook for each stage. |
| [Harmonic oscillator: three linearizations](../notebooks/examples/sho_kinetic_energy.ipynb) | The three linearizations give the same kinetic energy, on the smallest problem that shows all three. |
| [KPP–Fisher in 1-D](../notebooks/examples/kpp_fisher_1d.ipynb) | 88 element calls assemble the matrix at any grid size, and a direct solve confirms it. |
| [Vocabulary and symbols](concepts.md) | Every term and symbol the other pages assume is defined here. |
| [API reference](api.md) | Every class and function is listed with what it takes and returns. |

```{toctree}
:caption: Getting started
:maxdepth: 1

Vocabulary and symbols <concepts>
Notebook index <../notebooks/README>
Contributing <../CONTRIBUTING>
```

```{toctree}
:caption: Tutorials
:maxdepth: 1

Overview: the five stages <../notebooks/tutorials/01_overview>
1. The ODE <../notebooks/tutorials/02_ode>
2. Linearization <../notebooks/tutorials/03_linearization>
3. Bases <../notebooks/tutorials/04_basis/index>
4. The matrix <../notebooks/tutorials/05_matrix>
5. Vectors <../notebooks/tutorials/06_vector>
Boundary conditions <../notebooks/tutorials/07_boundary_conditions>
End-to-end solutions <../notebooks/tutorials/08_end_to_end>
Serialization <../notebooks/tutorials/09_serialization>
```

```{toctree}
:caption: Basic examples
:maxdepth: 1

Harmonic oscillator: three linearizations <../notebooks/examples/sho_kinetic_energy>
Koopman–von Neumann: energy conservation <../notebooks/examples/kvn_heisenberg>
Pivot switching: the logistic equation <../notebooks/examples/psc_logistic_switching>
```

```{toctree}
:caption: Advanced examples
:maxdepth: 1

KPP–Fisher, 1-D <../notebooks/examples/kpp_fisher_1d>
KPP–Fisher, 2-D <../notebooks/examples/kpp_fisher_2d>
KPP–Fisher, position-dependent 1-D <../notebooks/examples/kpp_fisher_position_dependent_1d>
KPP–Fisher, position-dependent 2-D <../notebooks/examples/kpp_fisher_position_dependent_2d>
Pivot switching: KPP–Fisher (Fig. 3) <../notebooks/examples/psc_fig3_kpp_fisher>
Pivot switching: phase field (Fig. 4) <../notebooks/examples/psc_fig4_phase_field>
```

```{toctree}
:caption: Reference
:maxdepth: 2

api
```

:::{only} internal
```{toctree}
:caption: Internal
:maxdepth: 1

Specification <internal/spec>
Conformance <_generated/examples/conformance>
Verification <internal/verification>
Design <internal/design>
Algorithms <internal/algorithms>
```

:::
