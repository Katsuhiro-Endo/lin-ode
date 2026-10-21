"""Regenerate ``notebooks/figures/five_stages.png``, the map on the overview page.

Run it after changing the pipeline's classes or their names. The output path is resolved
from this file, so the working directory does not matter.

The figure renders inside the 46em body column, so its apparent text size is
``fontsize * 736 / (figsize * 72)`` and a label fits its box when
``fontsize <= box_width_pt / chars``. Combining the two cancels the figure size: what
decides readability is the ratio of box width to label length, not the scale. So keep
each label short and give it a wide enough box, rather than enlarging the font.
"""

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INPUT, CLASS, CHOICE = ("#eef1f4", "#8d97a3"), ("#dbe8f6", "#3f6f9f"), ("#fdefd2", "#bf9138")

TITLE_SIZE, TOUCH_SIZE, NOTE_SIZE = 10.0, 9.6, 9.2

#         class                       cx   cy    w   style   what you touch        tutorial
NODES = [
    ("$f(u)$",                        14,  82,  26, INPUT,  "your equation",       ""),
    ("OrdinaryDifferentialSystem",     58,  82,  52, CLASS,  ".equation  .shape",   "02_ode"),
    ("LinearOperator",                108,  82,  32, CLASS,  "basis-free",          "03_linearization"),
    ("OperatorMatrix",                150,  82,  32, CLASS,  "matrix[bra, ket]",    "05_matrix"),
    ("LinearizationMethod",            76,  46,  38, CHOICE, "Koopman\nLiouville\nKoopmanVonNeumann", ""),
    ("TruncatedBasisSet",             132,  46,  44, CHOICE, "CarlemanBasis\nPSCarlemanBasis\nGaussianHermiteBasis", "04_basis"),
    ("$u(0)$  or  $g(u)$",             14,  12,  34, INPUT,  "initial value\nobservable", ""),
    ("Vector",                        150,  12,  32, CLASS,  "vector[state]",       "06_vector"),
]
# ``ly`` lifts a label clear of its neighbour: the boxes are wide, so the gap between two
# of them is narrower than either label. Negative puts the label under the arrow.
#        from to  label                       curve  exit  ly
EDGES = [
    (0, 1, "from_sympy\nfrom_strings",          0.00,   0,   2),
    (1, 2, ".create_operator(ode)",             0.00,   0,   2),
    (2, 3, "OperatorMatrix(basis, op)",         0.00,   0,   8),
    (4, 2, "",                                 -0.25,   0,   0),
    (5, 3, "",                                 -0.30, -14,   0),
    (5, 7, "",                                  0.30,  14,   0),
    (6, 7, "Vector(expr, basis, u)",            0.00,   0,  -5),
]

# A box is one title band plus one band per line of ``touch``, so a three-line choice box
# grows downwards instead of squeezing its text.
HEIGHTS = [11 + 6.2 * (touch.count("\n") + 1) for *_, touch, _ in NODES]

fig, ax = plt.subplots(figsize=(8.4, 5.3))
ax.set(xlim=(-6, 172), ylim=(-11, 104))
ax.axis("off")

for (name, cx, cy, w, (fill, edge), touch, tutorial), h in zip(NODES, HEIGHTS):
    top = cy + h / 2
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=3",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(cx, top - 5, name, ha="center", va="center", fontsize=TITLE_SIZE, weight="bold", color="#1c2a38")
    ax.text(cx, top - 10 - 3.1 * (touch.count("\n") + 1), touch, ha="center", va="center",
            fontsize=TOUCH_SIZE, color="#5a6675", family="monospace", linespacing=1.35)
    if tutorial:  # bottom-left, clear of the arrows leaving the box
        ax.text(cx - w / 2, cy - h / 2 - 4.5, tutorial, ha="left", va="center",
                fontsize=NOTE_SIZE, style="italic", color=edge)

for src, dst, label, curve, dx, ly in EDGES:
    (_, x0, y0, w0, _, _, _), (_, x1, y1, w1, _, _, _) = NODES[src], NODES[dst]
    h0, h1 = HEIGHTS[src], HEIGHTS[dst]
    if y0 == y1:  # along a track: box edge to box edge
        start, end = (x0 + w0 / 2, y0), (x1 - w1 / 2, y1)
    else:  # between tracks: box top/bottom, offset so two arrows never share an exit
        up = y1 > y0
        start = (x0 + dx, y0 + (h0 / 2 if up else -h0 / 2))
        end = (x1 + dx / 3, y1 + (-h1 / 2 if up else h1 / 2))
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#6b7684",
        connectionstyle=f"arc3,rad={curve}", shrinkA=1, shrinkB=1))
    if label:
        above = ly >= 0
        ax.text((start[0] + end[0]) / 2, y0 + (h0 / 2 + ly if above else -h0 / 2 + ly), label,
                ha="center", va="bottom" if above else "top",
                fontsize=NOTE_SIZE, color="#3f4a57", family="monospace")

ax.text(-6, 97, "operator track", fontsize=TOUCH_SIZE, style="italic", color="#8d97a3")
ax.text(-6, 30, "vector track", fontsize=TOUCH_SIZE, style="italic", color="#8d97a3")
fig.tight_layout()
OUT = pathlib.Path(__file__).resolve().parents[2] / "figures" / "five_stages.png"
fig.savefig(OUT, dpi=200, bbox_inches="tight")
