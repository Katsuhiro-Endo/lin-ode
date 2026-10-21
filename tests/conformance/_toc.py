"""Report `<h2>`/`<h3>` titles and order for the sections below this directory (read by
tests/testing.py's `section_title_and_order`/`file_title_and_order`).

Keyed by directory name for the `<h2>`, then by file basename for the `<h3>`s inside it;
order is dict position. An unlisted directory/file falls back to an auto-generated
title, sorted after everything below.
"""

TOC = {
    "linearized_operator": (
        "Linearized Operator",
        {
            "test_koopman_across_bases.py": "Koopman",
            "test_liouville_across_bases.py": "Liouville",
            "test_kvn_across_bases.py": "KvN",
        },
    ),
    "initial_state": (
        "Initial State",
        {
            "test_fixed_point.py": "Fixed Point",
            "test_distribution.py": "Distribution",
        },
    ),
    "objective_function": (
        "Objective Function",
        {
            "test_vector_representation.py": "Vector Representation",
            "test_matrix_representation.py": "Matrix Representation",
        },
    ),
    "json_serialization": (
        "JSON Serialization",
        {
            "test_round_trip.py": "JSON Round-Trip",
        },
    ),
    "analytic": (
        "Analytic Derivation",
        {
            "test_analytic_derivation.py": "Closed-Form Agreement",
        },
    ),
}
