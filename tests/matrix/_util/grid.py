"""The grid every case runs on, and the two sets of variables written over it.

``u[i]`` is what the library takes; ``SYM`` is the brute force's own, one plain symbol
per site, so neither can be mistaken for the other.
"""

import sympy as sp

from tests.util import brute_force
from tests.util.brute_force import Site

D, Q_0, Q_1, R_0, R_1, h = sp.symbols("D Q_0 Q_1 R_0 R_1 h", real=True)
u = sp.IndexedBase("u")

N_SITE = 4
GRID: list[Site] = list(range(N_SITE))
IDX_4 = sp.Idx("i", (0, N_SITE - 1))
DEGREE = 2

SYM = brute_force.make_site_symbols(GRID)
"""The brute force's own variables, one per site, kept apart from ``u[i]``."""
