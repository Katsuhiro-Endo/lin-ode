"""``BasisFunction`` canonicalises its ``(site, power)`` factors on construction."""

import sympy as sp
from pytest import mark

from lin_ode.basis import BasisFunction

i = sp.Idx("i")
factor = sp.Tuple


@mark.parametrize(
    "given_factors, expected_args",
    [
        (((2, 1), (1, 3)), (factor(1, 3), factor(2, 1))),
        (((1, 0), (2, 5)), (factor(2, 5),)),
        (((1, 0),), ()),
        (((i, 2), (1, 3)), (factor(1, 3), factor(i, 2))),
        (
            ((factor(1, 0), 1), (factor(0, 1), 2), (factor(0, 0), 3)),
            (
                factor(factor(0, 0), 3),
                factor(factor(0, 1), 2),
                factor(factor(1, 0), 1),
            ),
        ),
        (
            ((factor(i, 0), 1), (factor(0, 0), 2)),
            (factor(factor(0, 0), 2), factor(factor(i, 0), 1)),
        ),
    ],
    ids=[
        "sorted-by-site",
        "power-zero-dropped",
        "all-powers-zero-is-the-vacuum",
        "concrete-site-before-symbolic",
        "grid-sites-sorted-row-major",
        "concrete-grid-site-before-symbolic",
    ],
)
def test_sorts_factors_by_site_and_drops_zero_powers(
    given_factors: tuple, expected_args: tuple[sp.Tuple, ...]
) -> None:
    """Zero powers are dropped (u**0 = 1); concrete sites sort before symbolic."""
    assert BasisFunction(*given_factors).args == expected_args
