"""What ``test_coeff.py``'s region sweep needs, one module per job.

| module | holds |
| --- | --- |
| ``grid`` | the grid, and the two sets of variables written over it |
| ``cases`` | the regions and right-hand sides, and one ``Case`` resolved from them |
| ``from_lin_ode`` | what the library answers for a case, or its refusal |
| ``from_brute_force`` | the same entry, enumerated site by site |

Every function below ``resolve_case`` takes a ``Case``, so the lookup tables are read once.
"""

from .cases import PIVOTS, PROBED, REGION_SHAPES, RIGHT_HAND_SIDES, Case, resolve_case
from .from_brute_force import read_brute_force_entry
from .from_lin_ode import DECLINED, ask_lin_ode, expand_over_sites

__all__ = [
    "DECLINED",
    "PIVOTS",
    "PROBED",
    "REGION_SHAPES",
    "RIGHT_HAND_SIDES",
    "Case",
    "resolve_case",
    "ask_lin_ode",
    "read_brute_force_entry",
    "expand_over_sites",
]
