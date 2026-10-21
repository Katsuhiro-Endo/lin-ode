# API reference

Generated from the docstrings, in pipeline order. Names that appear in `lin_ode.__all__` are also
importable from the top-level package, for example `from lin_ode import OrdinaryDifferentialSystem`.

## The ODE

```{eval-rst}
.. automodule:: lin_ode.ordinary_differential_system
   :members:
   :show-inheritance:
```

## Linearization

```{eval-rst}
.. automodule:: lin_ode.operators
   :members:
   :show-inheritance:
```

## Bases

```{eval-rst}
.. automodule:: lin_ode.basis
   :members:
   :show-inheritance:
```

## Matrices

```{eval-rst}
.. automodule:: lin_ode.matrix
   :members:
   :show-inheritance:
```

## Vectors

```{eval-rst}
.. automodule:: lin_ode.vector
   :members:
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: lin_ode.observable
   :members:
   :show-inheritance:
```

## Expressions

```{eval-rst}
.. automodule:: lin_ode.indexed_expression
   :members:
   :show-inheritance:
```

## Type aliases

The names that appear in the signatures above.

```{eval-rst}
.. autodata:: lin_ode.vector.StructuredIdx
   :no-value:

.. autodata:: lin_ode.matrix.MatrixElement
   :no-value:

.. autoclass:: lin_ode.operators.IndexSpec
```

- ``StructuredIdx`` — one ``(site, power)`` factor, or a tuple of them.
- ``MatrixElement`` — ``complex | sympy.Expr``.
- ``IndexSpec`` — the contravariant type variable a :class:`~lin_ode.operators.Basis` is generic in.

