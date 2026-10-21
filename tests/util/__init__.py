"""What more than one test directory needs, so no directory owns another's tools.

| module | holds | | --- | --- | | ``equations`` | the right-hand sides tests drive, one
function each | | ``comparison`` | symbolic comparison and the pivot symbols | |
``rhs_at_sites`` | any ODE's right-hand side as a polynomial per grid site | |
``brute_force`` | a matrix entry from those polynomials, no lin_ode import |

``brute_force`` is the one independent answer every test compares against. It imports
nothing from lin_ode, so agreement is evidence rather than a restatement.
"""

from lin_ode.operators import Koopman, KoopmanVonNeumann, Liouville

METHODS = {
    "Koopman": Koopman(),
    "Liouville": Liouville(),
    "KvN": KoopmanVonNeumann(),
}
"""Name to method, in the column order of every entry table in the report."""
