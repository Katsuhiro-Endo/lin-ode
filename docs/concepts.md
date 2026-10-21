# Vocabulary and symbols

The terms and symbols the other pages assume. Anything used on one page only is defined on that
page.

## Vocabulary

### General terms

| term | symbol | one line |
|---|---|---|
| **linearization** | | replacing $\mathbf F$ by a linear operator acting on *functions* of the state |
| **site**, **(state variable) index** | $i$, $s$ | one point of the discretised grid, and the index that names it. On more than one axis the index is a **multi-index** $(i,j,\dots)$ |
| **state** | $u$ | what the ODE solves for, one component $u_s$ per site. Unqualified, **state** always means $u$; a member of the basis is a **basis state** |
| **density** | $\rho(u)$ | a distribution over states, the alternative to a fixed initial value. `Liouville` transports it, and KvN factors it as $\rho=\psi^*\psi$. The specification calls it a distribution |
| **stencil** | | the fixed neighbour offsets one term reads |
| **wrap** | | how an index outside its range resolves: an index running from $a$ over $n$ sites maps $i$ to $a+(i-a)\bmod n$. Every edge wraps unless a condition says otherwise |
| **region** | | the subset of the grid a condition applies on: `Facet` pins one axis, `Box` gives each axis a range, `&` intersects two pinned regions |
| **condition** | | a rule on a region, passed through `conditions`. A **state condition** holds $u$ at a value (`DirichletCondition`); a **derivative condition** sets $du/dt$ instead (`DerivativeCondition`) |
| **basis** | $\boldsymbol\Phi$ | the functions the state is expanded in |
| **truncation degree** | $K$, $P$ | the largest total degree the basis keeps; `degree` in the code |
| **generator** | $L$ | the operator driving the evolution |
| **acting site** | $s$ | the site a generator term is summed at, carrying $f_s$ and $\partial_s$. The term reads the equation's coefficients at $s$, not at the neighbours $f_s$ reaches |
| **observable** | $g(u)$ | a function of the state you read out |
| **expectation value** | $\langle g\rangle$ | an observable averaged over the density, $\int g\,\rho\,du$ |
| **projector** | | what turns one factor of an expression into a coefficient on a basis. A basis owns its projectors, and `Vector` uses the first that accepts the factor |
| **block** | | the submatrix on a chosen subset of basis states |
| **bra / ket** | $\langle\beta\rvert$, $\lvert\alpha\rangle$ | the two basis states naming a matrix element |
| **coupling** | $M[\beta,\alpha]$ | one $(\text{bra},\text{ket})$ pair, so one matrix element |
| **vacuum** | $\lvert\mathrm{vac}\rangle$ | the constant function $1$, the basis state with no factors; written `()` as a label |
| **structured (basis) index** | | a basis state as its `(site, power)` tuples, which is how you address one. A *basis* index names a basis function, where a *state variable* index names a site |
| **flat (basis) index** | | the same basis state as its integer position in $\boldsymbol\Phi$ |

### Linearizations

| term | symbol | one line |
|---|---|---|
| **Koopman** | $\mathcal K$ | transports the observable; the state is fixed. The Heisenberg picture |
| **Liouville** | $\mathcal L$ | transports the probability density; the observable is fixed. The Schrödinger picture |
| **Koopman–von Neumann** (KvN) | $\mathcal L_{\rm KvN}$ | transports an amplitude $\psi$ with $\rho=\psi^*\psi$; the observable becomes an operator, and a matrix once a basis is chosen |

### Bases

| term | symbol | one line |
|---|---|---|
| **Carleman** (CL) | $y_{\boldsymbol\alpha}$ | the basis of monomials $\prod_s u_s^{\alpha_s}$: it removes the nonlinearity by enlarging the state |
| **pivot-switching Carleman** (PSC) | | Carleman, but a monomial above the degree is re-expanded about a **pivot** instead of dropped |
| **pivot** | $s_i$ | the reference point a monomial above the degree is Taylor-expanded about |
| **fold** | | this project's name for PSC's re-expansion: replacing an above-degree monomial by its Taylor polynomial about the pivot, so the monomial returns as a sum over the states the basis keeps |
| **Gauss–Hermite** (GH) | $\psi_n$ | the basis of Hermite functions. Square-integrable, so `Liouville` and `KoopmanVonNeumann` accept it, and orthonormal, so a coefficient is an inner product |
| **$L^2$ basis** | | a basis of square-integrable functions on $\mathbb{R}^N$: the `is_l2` flag. `Liouville` and `KoopmanVonNeumann` need one, `Koopman` does not |
| **`IncompatibleBasisError`** | | what `OperatorMatrix` raises for a linearization that needs an $L^2$ basis and did not get one |

## Symbols

One symbol per quantity, used the same way in every notebook.

| symbol | is |
|---|---|
| $N$ | number of grid sites, so the state dimension |
| $n$ | sites per axis, what a basis is constructed from; $N=\prod_a n_a$ |
| $d$ | number of grid axes |
| $h$ | grid spacing |
| $i$, $j$ | the grid indices, matching the `sp.Idx` names in code: $u_i$, or $u_{i,j}$ on two axes. Used where a stencil is written out |
| $s$ | a site where the formula is about any site rather than a written-out stencil, as in $\sum_s f_s\partial_s$. In pivot-switching contexts $s_i$ is instead the pivot at site $i$, and only the pivot carries a subscript |
| $K$ | truncation degree |
| $P$ | the same degree in pivot-switching contexts, following [arXiv:2411.16233](https://arxiv.org/abs/2411.16233) |
| $u$, $f_s$ | the state, and its right-hand side at site $s$ |
| $\mathbf x$, $\mathbf F$ | the same state and right-hand side written as one $N$-vector rather than indexed by site, $\dot{\mathbf x}=\mathbf F(\mathbf x)$. The specification's notation |
| $g(u)$ | an observable, a function of the state; $\langle g\rangle$ is its expectation value |
| $\rho$, $\psi$ | the probability density, and the KvN amplitude with $\rho=\psi^*\psi$. Bold $\boldsymbol\rho$, $\boldsymbol\psi$ are their coefficient vectors on the basis. $\psi_n$ with an integer subscript is instead the $n$-th Hermite function |
| $\boldsymbol\alpha$, $\alpha_s$ | a basis state, and its power at site $s$; $\lvert\boldsymbol\alpha\rvert=\sum_s\alpha_s$ is the total degree |
| $y_{\boldsymbol\alpha}=\prod_s u_s^{\alpha_s}$ | the basis monomial, the same object as $\lvert\boldsymbol\alpha\rangle$ |
| $\mathbf e_s$ | one unit of power at site $s$ |
| $\phi_{\boldsymbol\alpha}$ | one basis function, written $\phi_b$ in the specification |
| $\boldsymbol\Phi$ | the basis functions as one vector, flat order, one block per total degree; $\Phi_{\boldsymbol\alpha}$ is its component |
| $\vec c$ | an observable's coefficients on the basis, so $g=\vec c^{\;\mathsf T}\boldsymbol\Phi$ |
| $\mathcal K=\sum_s f_s\partial_s$ | the **Koopman** operator; the reports write it $L_{\mathrm K}$ |
| $\mathcal L=\mathcal K^{\dagger}=-\sum_s\partial_s f_s$ | the **Liouville** operator, the adjoint of $\mathcal K$; the reports write it $L_{\mathrm L}$ |
| $\mathcal L_{\rm KvN}=-\tfrac12\sum_s\bigl(\partial_s f_s+f_s\partial_s\bigr)$ | the **KvN** operator; it transports a state, as $\mathcal L$ does; the reports write it $L_{\mathrm{KvN}}$ |
| $M$ | the matrix of an operator in a basis, $M[\beta,\alpha]=\langle\beta\rvert\cdot\lvert\alpha\rangle$ |
| $M_{\mathcal K}$ | the same, with a subscript naming *which* operator; row is the bra, column the ket |
| $A$ | the matrix that propagates the vector you actually solve for |
| $L$ | whichever of $\mathcal K,\mathcal L,\mathcal L_{\rm KvN}$ the context uses; in the KPP–Fisher notebooks $L=\mathcal K=\sum_s f_s\partial_s$ |

### When to transpose

`OperatorMatrix` always indexes $M$ with $M[\beta,\alpha]$ the coefficient of $\beta$ in
$\cdot\lvert\alpha\rangle$. What you propagate decides whether $A$ is $M$ or $M^{\mathsf T}$:

| you are evolving | $A$ | why |
|---|---|---|
| basis functions at a point, $\boldsymbol\Phi(\mathbf x(t))$, the Koopman read-out | $M^{\mathsf T}$ | $\mathcal K\phi_\alpha=\sum_\beta M[\beta,\alpha]\phi_\beta$, so $\dot\phi_\alpha$ sums down a *column* |
| coefficients, $\boldsymbol\rho(t)$ or $\vec c$, under Liouville or for an observable | $M$ | $\dot\rho_\beta=\sum_\alpha M[\beta,\alpha]\rho_\alpha$ sums along a *row* |

Get it backwards and the answer is silently wrong, so this is the one convention worth memorising.
