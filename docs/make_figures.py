# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# pyright: reportAttributeAccessIssue=false
# sympy stubs type IndexedBase subscripting and .args as Basic, so u[i - 1] and the
# from_sympy arguments below do not read as Indexed/Idx; the mismatches are noise here
# (same suppression as lin_ode/matrix/_coeff.py).
"""Draw the assembly-algorithm figures into notebooks/figures/*.png.

Two examples decide which matrix entries can be nonzero and which of them must share a
value. Both are combinatorial and hard to read off the code, so each gets a picture. The
plots live here, not in the notebooks, so the notebooks stay about the problem.

    uv run python docs/make_figures.py

The notebooks reference the PNGs by relative path and hold their own captions, so this
script only redraws images -- it never edits a notebook. Run it after changing
`iter_neighbours`, `find_reachable_bras`, `classify_by_shift`, `classify_by_zone`, `build`, or
`PSCarlemanBasis.truncation_rule`.
"""

import ast
import json
import pathlib
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Circle, Rectangle

from lin_ode import OrdinaryDifferentialSystem
from lin_ode.basis import CarlemanBasis, PSCarlemanBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES, FIGURES = ROOT / "notebooks" / "examples", ROOT / "notebooks" / "figures"
LABEL, TICK, TITLE, CELL = 14.0, 12.5, 14.5, 11.5
CODE_COLOUR, PROSE_COLOUR = "#9c4221", "0.15"
CALL_FACE, HIT_ALPHA = "#f6e7c1", 0.55
BRANCH = {"itself": ("#d9dde2", "0.35"), "hop": ("#bcd4ea", "#2f6494"),
          "raise": ("#c6e0c2", "#3d7a38")}


# ---------------------------------------------------------------------------- plumbing
DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign,
               ast.AnnAssign)


def notebook_namespace(name: str, needed: set[str]) -> dict:
    """Run a notebook's definitions and hand back what they defined.

    Only definitions: the notebooks interleave demo runs that take minutes. Executing the
    notebook's own functions is the point -- a figure must never show a copy of them.
    """
    namespace: dict = {}
    for n, cell in enumerate(json.loads((EXAMPLES / name).read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body if isinstance(node, DEFINITIONS)]
        exec(compile(tree, f"{name}:{n}", "exec"), namespace)  # noqa: S102
        if needed <= namespace.keys():
            return namespace
    raise LookupError(f"{name} never defined {sorted(needed - namespace.keys())}")


def save(fig, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"{stem}.png"
    fig.savefig(out, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")


def rich_text(ax, xy, text, size, colour=PROSE_COLOUR, box_alignment=(0.0, 0.5),
              coords="data"):
    """A label whose `backticked` spans come out monospaced and coloured."""
    parts = []
    for n, seg in enumerate(text.split("`")):
        if seg:
            code = n % 2 == 1
            parts.append(TextArea(seg, textprops={
                "fontsize": size - 0.4 * code, "family": "monospace" if code else "sans-serif",
                "color": CODE_COLOUR if code else colour}))
    ax.add_artist(AnnotationBbox(HPacker(children=parts, align="baseline", pad=0, sep=0),
                                 xy, xycoords=coords, frameon=False, pad=0,
                                 box_alignment=box_alignment, annotation_clip=False))


def rich_lines(ax, xy, lines, size, colour=PROSE_COLOUR, box_alignment=(0.0, 1.0),
               coords="axes fraction", sep=4.0):
    """A block of such labels, one per line."""
    rows = []
    for line in lines:
        parts = []
        for n, seg in enumerate(line.split("`")):
            if seg:
                code = n % 2 == 1
                parts.append(TextArea(seg, textprops={
                    "fontsize": size - 0.4 * code,
                    "family": "monospace" if code else "sans-serif",
                    "color": CODE_COLOUR if code else colour}))
        rows.append(HPacker(children=parts or [TextArea(" ")], align="baseline", pad=0, sep=0))
    ax.add_artist(AnnotationBbox(VPacker(children=rows, align="left", pad=0, sep=sep),
                                 xy, xycoords=coords, frameon=False, pad=0,
                                 box_alignment=box_alignment, annotation_clip=False))


def light_palette() -> list:
    """Distinct pale fills, enough that no two classes ever share one.

    Colour is the cue a reader trusts first, so a wrapped palette would claim two
    different classes are the same value.
    """
    return [c for name in ("Pastel1", "Pastel2", "Set3") for c in plt.get_cmap(name).colors]


def colours_for(tags: dict) -> dict:
    palette = light_palette()
    if len(tags) > len(palette):
        raise ValueError(f"{len(tags)} classes exceeds {len(palette)} distinct colours")
    return {key: palette[n] for n, key in enumerate(tags)}


def letters_by_size(cells: dict, offset: int = 0) -> dict:
    """A short tag per equivalence class, biggest class first, so cells can be read.

    `offset` lets two panels of one figure use disjoint letters -- the same letter meaning
    two different values across panels is worse than a longer alphabet.
    """
    counts: dict = {}
    for key in cells.values():
        counts[key] = counts.get(key, 0) + 1
    order = sorted(counts, key=lambda k: (-counts[k], str(k)))
    return {key: string.ascii_lowercase[(n + offset) % 26] for n, key in enumerate(order)}


def monomial(state) -> str:
    if not state:
        return "$1$"
    return "$" + "".join(f"u_{{{int(s)}}}" + ("" if int(p) == 1 else f"^{{{int(p)}}}")
                         for s, p in state) + "$"


def bare_monomial(state) -> str:
    if not state:
        return "1"
    return "".join(f"u_{{{int(s)}}}" + ("" if int(p) == 1 else f"^{{{int(p)}}}")
                   for s, p in state)


def element(bra, ket, value) -> str:
    return (rf"$\langle {bare_monomial(bra)}|\,L\,|{bare_monomial(ket)}\rangle"
            rf" = {sp.latex(value)}$")


def short(value) -> str:
    return f"${sp.latex(sp.nsimplify(value))}$"


def plain_state(state):
    """((coordinates, power), ...) back to ((site, power), ...) for labelling."""
    return tuple((site[0], power) for site, power in state)


# --------------------------------------------------------- figure: reach on the lattice
def tag_reach(ns, ket, shape, degree) -> tuple[set, dict]:
    """Every bra `find_reachable_bras` returns, with the branch that produced it.

    The function returns a bare set, so provenance is re-derived by replaying the same
    three branches through the notebook's own `iter_neighbours` -- the grid edge is
    decided in exactly one place.
    """
    def normalise(state):
        return tuple(sorted((ns["coord"](site), int(power)) for site, power in state))

    bras = ns["find_reachable_bras"](ket, shape, degree)
    powers = {ns["coord"](site)[0]: int(power) for site, power in ket}
    total = sum(powers.values())
    tagged = {normalise(ket): "itself, the diagonal"}
    for site, power in powers.items():
        for near in ns["iter_neighbours"]((site,), shape):
            moved = dict(powers)
            moved[site] -= 1
            moved[near[0]] = moved.get(near[0], 0) + 1
            tagged[tuple(sorted(((s,), p) for s, p in moved.items() if p > 0))] = \
                f"hop {site} to {near[0]}"
        if total < degree:
            raised = dict(powers) | {site: power + 1}
            tagged[tuple(sorted(((s,), p) for s, p in raised.items() if p > 0))] = \
                f"raise at {site}"
    unlabelled = {normalise(b) for b in bras} - set(tagged)
    if unlabelled:
        raise AssertionError(f"branches explain no bra for {unlabelled}")
    return {normalise(b) for b in bras}, tagged


def branch_of(label: str) -> tuple:
    return BRANCH["itself" if label.startswith("itself") else
                  "hop" if label.startswith("hop") else "raise"]


def reach_on_the_lattice(sites: int = 6, degree: int = 2) -> None:
    ns = notebook_namespace("kpp_fisher_1d.ipynb",
                            {"find_reachable_bras", "iter_neighbours", "coord", "fisher_1d"})
    basis = ns["CarlemanBasis"](sites, degree)
    matrix = OperatorMatrix(basis, ns["Koopman"]().create_operator(ns["fisher_1d"](sites)))
    shape = (sites,)
    examples = [(((2, 1),), "an interior ket"),
                (((0, 1),), "a ket at the left edge"),
                (((1, 1), (2, 1)), "a ket already at the truncation degree")]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.2))
    for ax, (ket, caption) in zip(axes, examples):
        bras, tagged = tag_reach(ns, ket, shape, degree)
        ordered = sorted(bras, key=lambda b: (not tagged[b].startswith("itself"),
                                              tagged[b].startswith("raise"), tagged[b]))
        flat_ket = tuple((ns["coord"](s), int(p)) for s, p in ket)
        rows = [(flat_ket, "", "#f6e7c1", "#8a6d1f")]
        rows += [(b, tagged[b], *branch_of(tagged[b])) for b in ordered]

        for y, (state, label, face, edge) in enumerate(rows):
            powers = {site[0]: power for site, power in state}
            for x in range(sites):
                filled = x in powers
                ax.add_patch(Rectangle((x - 0.42, y - 0.36), 0.84, 0.72,
                                       facecolor=face if filled else "white",
                                       edgecolor=edge if filled else "0.8",
                                       lw=1.6 if filled else 0.7))
                if filled:
                    ax.text(x, y, str(powers[x]), ha="center", va="center",
                            fontsize=CELL + 0.5, fontweight="bold", color="0.1")
            ax.text(sites - 0.35, y, label, ha="left", va="center", fontsize=TICK - 0.5,
                    color=edge)
        ax.text(sites - 0.35, 0,
                f"the input ket  $|{bare_monomial(plain_state(flat_ket))}\\rangle$",
                ha="left", va="center", fontsize=TICK, color="#8a6d1f")
        for y, bra in enumerate(ordered, start=1):
            ax.text(sites + 2.2, y,
                    element(plain_state(bra), plain_state(flat_ket), matrix[bra, ket]),
                    ha="left", va="center", fontsize=TICK - 0.5, color="0.15")

        ax.set_xlim(-0.4, sites + 6.0)
        ax.set_ylim(len(rows) - 0.5, -0.7)
        ax.set_xticks(range(sites))
        ax.set_xticklabels([f"$u_{{{x}}}$" for x in range(sites)], fontsize=TICK)
        ax.set_yticks([])
        ax.set_title(f"{caption}: {len(bras)} bras", fontsize=TITLE - 0.5, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    rich_lines(fig.axes[0], (0.5, 2.02), ["How `find_reachable_bras` works"], TITLE + 2.0,
               box_alignment=(0.5, 0.0), coords=fig.axes[0].transAxes)
    fig.axes[0].text(0.0, 1.86, "each box is a grid site; the number in it is that site's power",
                     transform=fig.axes[0].transAxes, ha="left", va="center",
                     fontsize=TICK - 0.5, color="0.35")
    terms = [("itself", r"diagonal — the $(Q-2dD/h^2)\,u_s\partial_{u_s}$ terms"),
             ("hop", r"diffusion — the $D\,u_{s'}\partial_{u_s}$ term"),
             ("raise", r"reaction — the $R\,u_s^2\partial_{u_s}$ term")]
    for n, (branch, label) in enumerate(terms):
        face, edge = BRANCH[branch]
        y = 1.68 - n * 0.15
        fig.axes[0].add_patch(Rectangle((0.02, y - 0.055), 0.020, 0.11, facecolor=face,
                                        edgecolor=edge, lw=1.4,
                                        transform=fig.axes[0].transAxes, clip_on=False))
        fig.axes[0].text(0.055, y, label, transform=fig.axes[0].transAxes, ha="left",
                         va="center", fontsize=TICK - 0.5, color="0.25")
    fig.tight_layout(rect=(0, 0, 1, 0.845))
    save(fig, "reach-on-lattice")


# ----------------------------------------------------------------- figure: the symmetry
def shift_symmetry(radius: int = 1) -> None:
    ns = notebook_namespace("kpp_fisher_1d.ipynb",
                            {"classify_by_shift", "canonicalise_positions", "fisher_1d"})
    grid_sites = 24
    basis = ns["CarlemanBasis"](grid_sites, 3)
    matrix = OperatorMatrix(basis, ns["Koopman"]().create_operator(ns["fisher_1d"](grid_sites)))
    examples = [(((5, 1), (6, 1)), "two adjacent sites"),
                (((20, 1), (21, 1)), "the same pattern, 15 sites along"),
                (((5, 1), (7, 1)), "a gap of 2"),
                (((5, 1), (9, 1)), "a gap of 4"),
                (((5, 1), (6, 1), (7, 1)), "three adjacent sites")]

    rows = []
    for state, caption in examples:
        occupied = [int(site) for site, _ in state]
        relabel = ns["canonicalise_positions"]([(s,) for s in occupied], radius)[0]
        rows.append((occupied, [relabel[s] for s in occupied], caption,
                     ns["classify_by_shift"](state, state), matrix[state, state]))

    order = list(dict.fromkeys(row[3] for row in rows))
    key_colour = {key: light_palette()[n] for n, key in enumerate(order)}
    STRIDE, RIGHT, MAPSTO, CAPTION, ELEMENT = 1.35, 6.1, 4.6, 10.4, 15.4

    fig, ax = plt.subplots(figsize=(16.4, 4.6))
    for y, (occupied, canonical, caption, key, value) in enumerate(rows):
        face = key_colour[key]
        for rank, (site, position) in enumerate(zip(occupied, canonical)):
            left, right = rank * STRIDE, RIGHT + rank * STRIDE
            for x, text, fc, ec in ((left, site, "#f6e7c1", "#8a6d1f"),
                                    (right, position, face, "0.35")):
                ax.add_patch(Rectangle((x - 0.34, y - 0.3), 0.68, 0.6, facecolor=fc,
                                       edgecolor=ec, lw=1.5))
                ax.text(x, y, str(text), ha="center", va="center", fontsize=CELL + 0.5,
                        fontweight="bold", color="0.1")
            if rank:
                gap = site - occupied[rank - 1]
                capped = min(gap, radius + 1)
                ax.annotate("", (left - 0.36, y), (left - STRIDE + 0.36, y),
                            arrowprops={"arrowstyle": "<->", "color": "0.5", "lw": 0.9,
                                        "shrinkA": 0, "shrinkB": 0})
                ax.text(left - STRIDE / 2, y - 0.34,
                        f"gap {gap}" + (f" $\\to$ {capped}" if capped != gap else ""),
                        ha="center", va="bottom", fontsize=TICK - 1.0,
                        color="#b3202a" if capped != gap else "0.4",
                        fontweight="bold" if capped != gap else "normal")
        ax.text(MAPSTO, y, r"$\longmapsto$", ha="center", va="center", fontsize=16, color="0.3")
        ax.text(CAPTION, y, caption, ha="left", va="center", fontsize=TICK - 0.3, color="0.25")
        diagonal = tuple((s, 1) for s in occupied)
        ax.text(ELEMENT, y, element(diagonal, diagonal, value), ha="left", va="center",
                fontsize=TICK - 0.5, color="0.15")

    ax.text(0.0, -1.75, f"each box is a site index; any gap of {radius + 1} or more is recorded "
                        f"as {radius + 1}, since $L$ moves a power only {radius} site",
            ha="left", va="center", fontsize=TICK - 0.5, color="0.35")
    for x, text, colour in ((STRIDE, "sites, with each gap", "#8a6d1f"),
                            (RIGHT + STRIDE, "canonical positions", "0.3")):
        ax.text(x, -1.05, text, ha="center", va="center", fontsize=TICK, color=colour)
    for x, text in ((CAPTION, "same fill = same key"), (ELEMENT, "the element")):
        ax.text(x, -1.05, text, ha="left", va="center", fontsize=TICK, color="0.3")
    ax.set_xlim(-0.9, 22.6)
    ax.set_ylim(len(rows) - 0.35, -2.05)
    ax.axis("off")
    rich_lines(ax, (0.0, 1.06), [f"How `canonicalise_positions` works: "
                                 f"{len(rows)} couplings, {len(order)} keys"],
               TITLE + 2.0, box_alignment=(0.0, 0.0), coords=ax.transAxes)
    fig.tight_layout()
    save(fig, "shift-symmetry")


# -------------------------------------------------------------- figure: the build trace
def replay(basis, reach_of, key_of, matrix, columns) -> tuple[list, dict]:
    """Run the assembly loop in its own order, recording the branch each visit took.

    `columns` must be a prefix of the loop, or a cell would be marked a cache hit whose
    call happened off the picture.
    """
    value, visits = {}, []
    for ket_flat in range(len(basis)):
        ket = basis.structured_idx_of_flat_index(ket_flat)
        for bra in reach_of(ket):
            key = key_of(bra, ket)
            called = key not in value
            if called:
                value[key] = matrix[bra, ket]
            if ket_flat in columns:
                visits.append({"cell": (basis.flat_index_of_structured_index(bra), ket_flat),
                               "key": key, "called": called, "kept": value[key] != 0,
                               "step": len(visits) + 1})
    return visits, value


def build_trace(sites: int = 8, degree: int = 2) -> None:
    ns = notebook_namespace("kpp_fisher_1d.ipynb",
                            {"find_reachable_bras", "classify_by_shift", "coord", "fisher_1d"})
    basis = ns["CarlemanBasis"](sites, degree)
    matrix = OperatorMatrix(basis, ns["Koopman"]().create_operator(ns["fisher_1d"](sites)))
    shape, dim = ns["coord"](basis.n_vars), len(basis)

    def reach_of(ket):
        return ns["find_reachable_bras"](ket, shape, degree)

    columns = list(range(sites + 1))          # vacuum + every degree-1 ket: a true prefix
    visits, value = replay(basis, reach_of, ns["classify_by_shift"], matrix, columns)
    whole, _ = replay(basis, reach_of, ns["classify_by_shift"], matrix, range(dim))

    tags = letters_by_size({v["cell"]: v["key"] for v in visits})
    colours = colours_for(tags)
    rows = sorted({v["cell"][0] for v in visits})
    row_at = {r: n for n, r in enumerate(rows)}
    col_at = {c: n for n, c in enumerate(columns)}
    calls = sum(v["called"] for v in visits)
    dropped = sum(not v["kept"] for v in visits)

    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.06)
    ax, side = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    for v in visits:
        row, ket = v["cell"]
        x, y = col_at[ket], row_at[row]
        face = colours[v["key"]] if v["kept"] else "0.88"
        ax.add_patch(Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor=face if v["called"] else (*face[:3], HIT_ALPHA) if v["kept"] else face,
            hatch="////" if not v["kept"] else None,
            edgecolor="#1a1a1a" if v["called"] else "0.62",
            lw=2.4 if v["called"] else 0.8, zorder=3 if v["called"] else 2))
        ax.text(x, y, tags[v["key"]], ha="center", va="center", fontsize=CELL,
                fontweight="bold" if v["called"] else "normal", color="0.1", zorder=4)

    def total_degree(flat):
        return sum(int(p) for _, p in basis.structured_idx_of_flat_index(flat))

    for n in range(1, len(columns)):
        if total_degree(columns[n]) != total_degree(columns[n - 1]):
            ax.axvline(n - 0.5, color="0.45", lw=1.0, ls=(0, (4, 3)))
    for n in range(1, len(rows)):
        if total_degree(rows[n]) != total_degree(rows[n - 1]):
            ax.axhline(n - 0.5, color="0.45", lw=1.0, ls=(0, (4, 3)))

    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([monomial(basis.structured_idx_of_flat_index(c)) for c in columns],
                       fontsize=TICK, rotation=45, ha="right")
    ax.set_yticklabels([monomial(basis.structured_idx_of_flat_index(r)) for r in rows],
                       fontsize=TICK)
    ax.set_xlabel(r"ket $\alpha$   (in loop order)", fontsize=LABEL)
    ax.set_ylabel(r"bra $\beta$", fontsize=LABEL)
    ax.set_aspect("equal")
    rich_lines(ax, (0.0, 1.03), [f"How `build` works: {len(visits)} visits, {calls} calls"],
               TITLE + 2.0, box_alignment=(0.0, 0.0), coords=ax.transAxes)

    side.axis("off")
    states = [("white", None, "#1a1a1a", 2.4, calls, "call", "calls",
               "the key was new, so `matrix[bra, ket]` ran"),
              ((0.72, 0.78, 0.86, HIT_ALPHA), None, "0.62", 0.8, len(visits) - calls,
               "cache hit", "cache hits", "the key was already in `value`"),
              ("0.88", "////", "0.62", 0.8, dropped, "dropped entry", "dropped entries",
               "the value came out 0, so nothing was written")]
    for n, (face, hatch, edge, lw, count, one, many, why) in enumerate(states):
        y = 0.95 - n * 0.16
        side.add_patch(Rectangle((0.02, y - 0.045), 0.075, 0.075, facecolor=face, hatch=hatch,
                                 edgecolor=edge, lw=lw, transform=side.transAxes, clip_on=False))
        rich_text(side, (0.135, y + 0.005), f"{count} {one if count == 1 else many}",
                  TICK + 0.3, coords=side.transAxes)
        rich_text(side, (0.135, y - 0.05), why, TICK - 0.8, colour="0.35",
                  coords=side.transAxes)

    log = ["the calls, in the order they happen:", ""]
    for v in visits:
        if v["called"]:
            bra, ket = v["cell"]
            log.append(f"  {tags[v['key']]}    "
                       f"{monomial(basis.structured_idx_of_flat_index(ket))} $\\to$ "
                       f"{monomial(basis.structured_idx_of_flat_index(bra))}"
                       f"  = {short(value[v['key']])}")
    log += ["", "every later visit matches one of these",
            f"letters, so the remaining {len(visits) - calls} are free.", "",
            f"whole matrix: {len(whole)} visits, "
            f"{sum(v['called'] for v in whole)} calls,",
            f"against $\\dim^2 = {dim ** 2}$ pair by pair."]
    rich_lines(side, (0.02, 0.50), log, TICK - 0.8, coords=side.transAxes, sep=3.4)
    save(fig, "build-trace")


# ----------------------------------------------------------------- figure: the zone map
ZONE_FACE = {False: "#e6edf3", True: "#fbdcc4"}
ZONE_EDGE = {False: "#3f6c93", True: "#b3651f"}


def zone_rows(matrix) -> list:
    """The couplings the notebook prints, as (bra, ket, acting site, branch, caption)."""
    return [(((0, 1),), ((0, 1),), (0,), "itself", "diagonal, the site is outside"),
            (((2, 1),), ((2, 1),), (2,), "itself", "diagonal, the site is inside"),
            (((1, 1), (2, 1)), ((1, 1), (2, 1)), (1, 2), "itself",
             "diagonal, one site on each side"),
            (((1, 1),), ((2, 1),), (2,), "hop", r"hop $2\to1$, the source is inside"),
            (((2, 1),), ((1, 1),), (1,), "hop", r"hop $1\to2$, the source is outside"),
            (((2, 2),), ((2, 1),), (2,), "raise", "reaction at an inside site"),
            (((0, 2),), ((0, 1),), (0,), "raise", "reaction at an outside site")]


def zone_map(sites: int = 6, degree: int = 2, box=((2, 3),)) -> None:
    ns = notebook_namespace("kpp_fisher_position_dependent_1d.ipynb",
                            {"in_box", "fisher", "coord"})
    basis = ns["CarlemanBasis"](sites, degree)
    matrix = OperatorMatrix(basis, ns["Koopman"]().create_operator(ns["fisher"]((sites,), box)))
    zone_of = {s: ns["in_box"]((s,), box) for s in range(sites)}
    LEGEND = sites + 1.1

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(15.4, 7.0), gridspec_kw={"height_ratios": [1.0, 4.0], "hspace": 0.34})

    # ---- the zone map itself: which sites carry which coefficient set
    for s in range(sites):
        top.add_patch(Rectangle((s - 0.44, -0.38), 0.88, 0.76,
                                facecolor=ZONE_FACE[zone_of[s]],
                                edgecolor=ZONE_EDGE[zone_of[s]], lw=1.8))
        top.text(s, 0, f"$u_{{{s}}}$", ha="center", va="center", fontsize=CELL + 1.5,
                 color="0.1")
    lo, hi = box[0]
    top.plot([lo - 0.44, lo - 0.44, hi + 0.44, hi + 0.44], [0.62, 0.80, 0.80, 0.62],
             color=ZONE_EDGE[True], lw=1.6, clip_on=False)
    rich_text(top, (LEGEND, 0.30), r"inside — `zone(s)` is `True`:  " + "$D_1, Q_1, R_1$",
              TICK, colour=ZONE_EDGE[True])
    rich_text(top, (LEGEND, -0.30), r"outside — `zone(s)` is `False`:  " + "$D_0, Q_0, R_0$",
              TICK, colour=ZONE_EDGE[False])
    top.text((lo + hi) / 2, 0.95, f"`box = (({lo}, {hi}),)`", ha="center", va="bottom",
             fontsize=TICK, family="monospace", color=CODE_COLOUR)
    top.set_xlim(-0.7, sites + 8.4)
    top.set_ylim(-0.75, 1.35)
    top.axis("off")

    # ---- one row per printed coupling, drawn on the same chain
    BRA, VALUE, CAPTION = sites + 2.4, 2 * sites + 4.0, 2 * sites + 9.2

    def chain(ax, y, powers, ring, origin):
        for s in range(sites):
            filled = s in powers
            ax.add_patch(Rectangle((origin + s - 0.44, y - 0.32), 0.88, 0.64,
                                   facecolor=ZONE_FACE[zone_of[s]],
                                   alpha=1.0 if filled else 0.45,  # the zone band stays visible
                                   edgecolor=ZONE_EDGE[zone_of[s]] if filled else "0.86",
                                   lw=1.7 if filled else 0.7))
            if filled:
                ax.text(origin + s, y, str(powers[s]), ha="center", va="center", fontsize=CELL,
                        fontweight="bold", color="0.1")
        for s in ring:  # the site whose coefficients this term reads
            ax.add_patch(Rectangle((origin + s - 0.56, y - 0.44), 1.12, 0.88, facecolor="none",
                                   edgecolor=ZONE_EDGE[zone_of[s]], lw=1.5, ls=(0, (2, 1.6))))

    bottom.add_patch(Rectangle((-0.75, 2.45), CAPTION - 1.1, 2.10, facecolor="#f4f1ea",
                               edgecolor="none", zorder=0))  # rows 4-5: one bond, both ways
    for y, (bra, ket, acting, _, caption) in enumerate(zone_rows(matrix)):
        chain(bottom, y, dict(ket), acting, 0)
        chain(bottom, y, dict(bra), (), BRA)
        bottom.text(BRA - 1.3, y, r"$\longmapsto$", ha="center", va="center", fontsize=17,
                    color="0.35")
        bottom.text(VALUE, y, short(matrix[bra, ket]), ha="left", va="center", fontsize=TICK,
                    color="0.15")
        bottom.text(CAPTION, y, caption, ha="left", va="center", fontsize=TICK - 0.5,
                    color="0.3")
    for x, text in ((sites / 2 - 0.5, r"ket $|\alpha\rangle$"),
                    (BRA + sites / 2 - 0.5, r"bra $\langle\beta|$")):
        bottom.text(x, -1.0, text, ha="center", va="center", fontsize=LABEL, color="0.25")
    for origin in (0, BRA):  # site labels under each chain
        for s in range(sites):
            bottom.text(origin + s, (len(zone_rows(matrix)) - 1) + 0.85, f"$u_{{{s}}}$",
                        ha="center", va="center", fontsize=TICK - 0.5,
                        color=ZONE_EDGE[zone_of[s]])
    for x, text in ((VALUE, r"$\langle\beta|\,L\,|\alpha\rangle$"),
                    (CAPTION, "which term of $L$ acts")):
        bottom.text(x, -1.0, text, ha="left", va="center", fontsize=LABEL, color="0.25")
    bottom.set_xlim(-0.7, 2 * sites + 16.0)
    bottom.set_ylim(len(zone_rows(matrix)) + 0.15, -1.5)
    bottom.set_xticks([])
    bottom.set_yticks([])
    for spine in bottom.spines.values():
        spine.set_visible(False)
    bottom.tick_params(length=0)

    rich_lines(top, (0.0, 1.55), ["Which sites read which coefficients"], TITLE + 2.0,
               box_alignment=(0.0, 0.0), coords=top.transAxes)
    save(fig, "zone-map")


# ------------------------------------------------------------- figure: the zone classes
def touched(basis, reach_of, key_of, matrix, kets) -> tuple[list, dict, dict]:
    """The nonzero cells of the given columns, and the rows they actually occupy."""
    cells, value = {}, {}
    for ket_flat in kets:
        ket = basis.structured_idx_of_flat_index(ket_flat)
        for bra in reach_of(ket):
            key = key_of(bra, ket)
            if key not in value:
                value[key] = matrix[bra, ket]
            if value[key] != 0:
                cells[(basis.flat_index_of_structured_index(bra), ket_flat)] = key
    return sorted({r for r, _ in cells}), cells, value


def audit(basis, reach_of, key_of, matrix) -> tuple[int, int]:
    """How many classes the key makes, and how many hold more than one value.

    A class holding two values means one call's answer would be written into cells it does
    not belong to. This is the check that says whether a key may be used at all.
    """
    holds: dict = {}
    for ket_flat in range(len(basis)):
        ket = basis.structured_idx_of_flat_index(ket_flat)
        for bra in reach_of(ket):
            holds.setdefault(key_of(bra, ket), set()).add(matrix[bra, ket])
    return len(holds), sum(len(g) > 1 for g in holds.values())


def clash_in(basis, reach_of, key_of, matrix) -> tuple:
    """One key that holds two values, with two couplings that prove it.

    Printed so the example worked by hand in the notebook can be checked against the key
    the figure actually draws.
    """
    holds: dict = {}
    for ket_flat in range(len(basis)):
        ket = basis.structured_idx_of_flat_index(ket_flat)
        for bra in reach_of(ket):
            holds.setdefault(key_of(bra, ket), {})[matrix[bra, ket]] = (bra, ket)
    for seen in holds.values():
        if len(seen) > 1:
            return tuple(sorted(seen.items(), key=lambda kv: str(kv[0])))[:2]
    return ()


def zones_split_classes(sites: int = 10, degree: int = 2, box=((4, 7),)) -> None:
    ns = notebook_namespace("kpp_fisher_position_dependent_1d.ipynb",
                            {"find_reachable_bras", "classify_by_zone", "coord", "in_box",
                             "fisher"})
    basis = ns["CarlemanBasis"](sites, degree)
    matrix = OperatorMatrix(basis, ns["Koopman"]().create_operator(ns["fisher"]((sites,), box)))
    shape = ns["coord"](basis.n_vars)

    def reach_of(ket):
        return ns["find_reachable_bras"](ket, shape, degree)

    # the two keys the notebook itself builds with: a constant zone, and the real one
    keys = [(lambda b, k: ns["classify_by_zone"](b, k, lambda site: 0),
             "`classify_by_zone(bra, ket, lambda site: 0)`",
             "every site tagged alike — geometry only"),
            (lambda b, k: ns["classify_by_zone"](b, k, lambda s: ns["in_box"](s, box)),
             "`classify_by_zone(bra, ket, zone)`",
             "each site tagged with its own zone")]
    # sites 2..6: two outside the inclusion, then across its left edge, then two inside
    kets = [basis.flat_index_of_structured_index(((s, 1),)) for s in range(2, 7)]

    def label_of(flat):  # a state and the zone colour its single site sits in
        state = basis.structured_idx_of_flat_index(flat)
        return monomial(state), ZONE_EDGE[ns["in_box"]((int(state[0][0]),), box)]

    fig = plt.figure(figsize=(12.8, 6.6))
    grid = fig.add_gridspec(1, 4, width_ratios=[2.0, 1.05, 2.0, 1.05], wspace=0.30)
    for n, (key_of, name, subtitle) in enumerate(keys):
        rows, cells, value = touched(basis, reach_of, key_of, matrix, kets)
        classes, unsound = audit(basis, reach_of, key_of, matrix)
        tags = letters_by_size(cells, offset=0 if n == 0 else 12)
        colours = colours_for(tags)
        ax = fig.add_subplot(grid[0, 2 * n])
        row_at = {r: i for i, r in enumerate(rows)}
        col_at = {c: i for i, c in enumerate(kets)}
        for (row, ket), key in cells.items():
            x, y = col_at[ket], row_at[row]
            # a cell the cached value is wrong for: same tag, different true entry
            truth = matrix[basis.structured_idx_of_flat_index(row),
                           basis.structured_idx_of_flat_index(ket)]
            wrong = truth != value[key]
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=colours[key],
                                   edgecolor="#b3202a" if wrong else "0.25",
                                   lw=2.6 if wrong else 0.8, zorder=3 if wrong else 2))
            ax.text(x, y, tags[key], ha="center", va="center", fontsize=CELL,
                    fontweight="bold", color="0.1", zorder=4)
        ax.set_xlim(-0.5, len(kets) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(kets)))
        ax.set_yticks(range(len(rows)))
        for axis, flats, kwargs in ((ax.xaxis, kets, {"rotation": 45, "ha": "right"}),
                                    (ax.yaxis, rows, {})):
            texts, colour = zip(*(label_of(f) for f in flats))
            axis.set_ticklabels(texts, fontsize=TICK, **kwargs)
            for tick, c in zip(axis.get_ticklabels(), colour):
                tick.set_color(c)
        verdict = (f"{classes} keys, {unsound} of them holding two different values"
                   if unsound else f"{classes} keys, each holding a single value")
        rich_lines(ax, (0.0, 1.02), [name, subtitle, verdict], TITLE - 2.5,
                   colour="#b3202a" if unsound else "#1a6b3c",
                   box_alignment=(0.0, 0.0), coords=ax.transAxes, sep=5.0)
        ax.set_xlabel(r"ket $\alpha$", fontsize=LABEL)
        if n == 0:
            ax.set_ylabel(r"bra $\beta$", fontsize=LABEL)
        ax.set_aspect("equal")

        legend = fig.add_subplot(grid[0, 2 * n + 1])
        legend.axis("off")
        lines = ["one call per letter:", ""]
        lines += [f"{tag} = {short(value[key])}" for key, tag in tags.items()]
        rich_lines(legend, (0.02, 0.99), lines, TICK - 0.5, coords=legend.transAxes, sep=6.0)
        if unsound:
            legend.add_patch(Rectangle((0.02, 0.075), 0.13, 0.045, transform=legend.transAxes,
                                       facecolor="#f2f2f2", edgecolor="#b3202a", lw=2.6))
            rich_lines(legend, (0.02, 0.02),
                       ["that value is wrong", "in this cell"], TICK - 0.5,
                       colour="#b3202a", coords=legend.transAxes, sep=4.0)
    fig.subplots_adjust(top=0.82)
    def named(state):  # sites arrive bare from the basis, as tuples from find_reachable_bras
        return bare_monomial(tuple((ns["coord"](s)[0], p) for s, p in state))

    for val, (bra, ket) in clash_in(basis, reach_of, keys[0][0], matrix):
        print(f"    one shift-only key holds  <{named(bra)}|L|{named(ket)}>"
              f" = {sp.sstr(val)}")
    save(fig, "zone-classes")


# ------------------------------------------------------------------- figure: the PSC fold
def fold_weights(degree: int, raised: int, pivot) -> list:
    """What the PSC fold puts on each retained state, for `u**raised` truncated at `degree`."""
    basis = PSCarlemanBasis(1, degree, pivot)
    folded = sp.expand(basis.truncation_rule((0, raised)))
    states = [basis.state_type()] + [basis.state_type((0, p)) for p in range(1, degree + 1)]
    return [sp.simplify(folded.coeff(state)) for state in states]


def psc_fold(degree: int = 3) -> None:
    """The one operation that separates PSC from Carleman: weights, for a symbolic pivot."""
    s = sp.Symbol("s")
    rows = [degree + 1, degree + 2]  # one and two levels above the basis
    columns = ["$1$"] + [f"$u^{p}$" if p > 1 else "$u$" for p in range(1, degree + 1)]
    STRIDE, LEFT, CAPTION = 2.5, -3.2, 11.0

    fig, ax = plt.subplots(figsize=(9.6, 2.1))
    for y, raised in enumerate(rows):
        weights = fold_weights(degree, raised, s)
        source = f"$u^{raised}$" if raised > 1 else "$u$"
        ax.text(LEFT, y, f"{source} $\\longmapsto$", ha="left", va="center",
                fontsize=CELL + 1.5, color="0.15")
        for column, weight in enumerate(weights):
            zero = weight == 0
            face, edge = ("0.94", "0.80") if zero else (CALL_FACE, "#8a6d1f")
            ax.add_patch(Rectangle((column * STRIDE - 1.0, y - 0.32), 2.0, 0.64,
                                   facecolor=face, edgecolor=edge, lw=1.4))
            ax.text(column * STRIDE, y, f"${sp.latex(weight)}$", ha="center", va="center",
                    fontsize=CELL, color="0.7" if zero else "0.1",
                    fontweight="normal" if zero else "bold")

    for column, name in enumerate(columns):
        ax.text(column * STRIDE, -1.05, name, ha="center", va="center", fontsize=TICK,
                color="0.3")
    ax.text(LEFT, -1.05, "folds onto:", ha="left", va="center", fontsize=TICK,
            color="0.3")
    ax.set_xlim(LEFT - 0.4, CAPTION - 0.6)
    ax.set_ylim(len(rows) + 0.55, -1.45)
    ax.axis("off")
    rich_lines(ax, (0.0, 1.10), [f"Fold weights at truncation degree $P={degree}$"],
               TITLE + 2.0, box_alignment=(0.0, 0.0), coords=ax.transAxes)
    fig.tight_layout()
    save(fig, "psc-fold")


def psc_elements(degree: int = 3, sites: int = 4) -> None:
    """The two questions every assembly asks: which elements are nonzero, and which repeat."""
    s = sp.Symbol("s")
    x, i = sp.IndexedBase("x"), sp.Idx("i", (0, 0))
    logistic = OrdinaryDifferentialSystem.from_sympy(x[i] * (1 - x[i]), x, (i,))
    ket = ((0, degree),)
    bras = [()] + [((0, p),) for p in range(1, degree + 1)]

    def elements(basis) -> list:
        matrix = OperatorMatrix(basis, Koopman().create_operator(logistic))
        return [sp.simplify(sp.sympify(matrix[bra, ket])) for bra in bras]

    carleman = elements(CarlemanBasis(1, degree))
    psc = elements(PSCarlemanBasis(1, degree, pivot=s))
    grid = PSCarlemanBasis(sites, degree, pivot=s)
    folds = []
    for site in range(sites):
        folded = sp.expand(grid.truncation_rule((site, degree + 1)))
        states = [grid.state_type()] + [grid.state_type((site, p))
                                        for p in range(1, degree + 1)]
        folds.append([sp.simplify(folded.coeff(state)) for state in states])

    bra_labels = ["$\\langle 1|$"] + [
        (f"$\\langle x^{p}|$" if p > 1 else "$\\langle x|$")
        for p in range(1, degree + 1)
    ]
    STRIDE, LEFT = 2.6, -5.4
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13.4, 5.5), gridspec_kw={"height_ratios": [1.0, 1.15]}
    )

    def cells(ax, y, values, *, highlight=()) -> None:
        for column, value in enumerate(values):
            zero = value == 0
            new = column in highlight
            face = "0.94" if zero else CALL_FACE
            edge = "#b3202a" if new else ("0.80" if zero else "#8a6d1f")
            ax.add_patch(Rectangle((column * STRIDE - 1.05, y - 0.30), 2.1, 0.60,
                                   facecolor=face, edgecolor=edge,
                                   lw=2.2 if new else 1.4))
            ax.text(column * STRIDE, y, f"${sp.latex(value)}$", ha="center", va="center",
                    fontsize=CELL, color="0.7" if zero else "0.1",
                    fontweight="normal" if zero else "bold")

    # ---- which elements are nonzero
    turned_on = [c for c, (a, b) in enumerate(zip(carleman, psc)) if a == 0 and b != 0]
    for y, (label, values, hi) in enumerate([
        ("conventional Carleman", carleman, ()),
        ("PSC, any pivot $s$", psc, turned_on),
    ]):
        top.text(LEFT, y, label, ha="left", va="center", fontsize=TICK - 0.3, color="0.2")
        cells(top, y, values, highlight=hi)
    for column, name in enumerate(bra_labels):
        top.text(column * STRIDE, -0.95, name, ha="center", va="center", fontsize=TICK,
                 color="0.3")
    top.text(LEFT, -0.95, f"$|x^{degree}\\rangle$ reaches:", ha="left",
             va="center", fontsize=TICK, color="0.3")
    top.text(STRIDE * (degree + 0.55), 0,
             f"{sum(v != 0 for v in carleman)} nonzero", ha="left", va="center",
             fontsize=TICK - 0.5, color="0.35")
    top.text(STRIDE * (degree + 0.55), 1,
             f"{sum(v != 0 for v in psc)} nonzero: the fold fills the row", ha="left",
             va="center", fontsize=TICK - 0.5, color="#b3202a")
    top.set_xlim(LEFT - 0.4, STRIDE * (degree + 2.9))
    top.set_ylim(1.9, -1.35)
    top.axis("off")
    rich_lines(top, (0.0, 1.16),
               [f"Nonzero: $L\\,x^{degree}="
                f"{sp.latex(sp.expand(degree * sp.Symbol('x')**(degree - 1) * (sp.Symbol('x') - sp.Symbol('x')**2)))}$, "
                f"and $x^{degree + 1}$ is out of the basis at $P={degree}$"],
               TITLE, box_alignment=(0.0, 0.0), coords=top.transAxes)

    # ---- which elements repeat
    for y, weights in enumerate(folds):
        bottom.text(LEFT, y, f"$u_{y}^{degree + 1}\\longmapsto$", ha="left", va="center",
                    fontsize=CELL + 1.0, color="0.15")
        cells(bottom, y, weights)
        if y:
            bottom.text(STRIDE * (degree + 0.55), y, "same weights", ha="left",
                        va="center", fontsize=TICK - 0.5, color="0.35")
    bottom.text(STRIDE * (degree + 0.55), 0, "computed once", ha="left", va="center",
                fontsize=TICK - 0.5, color="#b3202a")
    for column, p in enumerate(range(degree + 1)):
        name = "$1$" if p == 0 else (f"$u_j^{p}$" if p > 1 else "$u_j$")
        bottom.text(column * STRIDE, -0.95, name, ha="center", va="center", fontsize=TICK,
                    color="0.3")
    bottom.text(LEFT, -0.95, "weights on:", ha="left", va="center", fontsize=TICK,
                color="0.3")
    bottom.set_xlim(LEFT - 0.4, STRIDE * (degree + 2.9))
    bottom.set_ylim(sites - 0.1, -1.35)
    bottom.axis("off")
    rich_lines(bottom, (0.0, 1.10),
               [f"Same value: the fold reads no site index, so all {sites} sites share "
                f"one element call"],
               TITLE, box_alignment=(0.0, 0.0), coords=bottom.transAxes)
    fig.tight_layout(h_pad=2.6)
    save(fig, "psc-elements")


def psc_column(degree: int = 3, sites: int = 8) -> None:
    """Where the fold actually fires, and how one call covers a whole shape."""
    pivot = sp.Symbol("s")
    u, i = sp.IndexedBase("u"), sp.Idx("i", (0, sites - 1))
    rhs = u[i - 1] - 2 * u[i] + u[i + 1] + u[i] * (1 - u[i])
    ode = OrdinaryDifferentialSystem.from_sympy(rhs, u, (i,))
    namespace = notebook_namespace("psc_fig3_kpp_fisher.ipynb", {"find_bras",
                                                                "find_reference_ket"})
    find_bras, reference = namespace["find_bras"], namespace["find_reference_ket"]

    def nonzeros(basis, ket) -> int:
        matrix = OperatorMatrix(basis, Koopman().create_operator(ode))
        labels = [basis.structured_idx_of_flat_index(k) for k in range(len(basis))]
        return sum(sp.simplify(sp.sympify(matrix[lab, ket])) != 0 for lab in labels)

    carleman = CarlemanBasis(sites, degree)
    psc = PSCarlemanBasis(sites, degree, pivot=pivot)
    kets = [((0, 1), (1, 1)), ((0, 1), (1, 2))]
    rows = [(k, nonzeros(carleman, k), nonzeros(psc, k), len(find_bras(k, degree)))
            for k in kets]

    shapes: dict = {}
    for flat in range(len(carleman)):
        ket = carleman.structured_idx_of_flat_index(flat)
        shapes.setdefault(reference(ket)[0], []).append(ket)
    family = shapes[kets[1]]

    def show(ket) -> str:
        if not ket:
            return "$1$"
        return "$" + "".join(f"u_{s}^{p}" if p > 1 else f"u_{s}" for s, p in ket) + "$"

    fig, (top, ring) = plt.subplots(1, 2, figsize=(13.0, 4.0),
                                    gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- nonzero: the fold fires only at the top degree
    HEAD = ["ket", "deg", "Carleman", "PSC", "`find_bras`"]
    AT = [0.0, 2.6, 5.4, 7.8, 10.4]
    for name, at in zip(HEAD, AT):
        top.text(at, -0.85, name.strip("`"), ha="center", va="center",
                 fontsize=TICK, color="0.3",
                 style="italic" if name.startswith("`") else "normal")
    for y, (ket, n_carleman, n_psc, bound) in enumerate(rows):
        total = sum(p for _, p in ket)
        fired = n_psc > n_carleman
        cells = [show(ket), f"${total}$", f"${n_carleman}$", f"${n_psc}$", f"${bound}$"]
        for column, (text, at) in enumerate(zip(cells, AT)):
            emphasis = fired and column == 3
            if column in (2, 3):
                top.add_patch(Rectangle((at - 0.62, y - 0.28), 1.24, 0.56,
                                        facecolor=CALL_FACE if emphasis else "0.95",
                                        edgecolor="#b3202a" if emphasis else "0.82",
                                        lw=2.2 if emphasis else 1.2))
            top.text(at, y, text, ha="center", va="center", fontsize=CELL,
                     color="0.1", fontweight="bold" if emphasis else "normal")
    top.set_xlim(-1.3, 11.7)
    top.set_ylim(len(rows) + 0.2, -1.25)
    top.axis("off")
    rich_lines(top, (0.0, 1.10), ["Nonzero entries in one column"],
               TITLE - 0.5, box_alignment=(0.0, 0.0), coords=top.transAxes)

    # ---- same value: one call per shape
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, sites, endpoint=False)
    powers = dict(kets[1])
    for site, angle in enumerate(angles):
        x, y = np.cos(angle), np.sin(angle)
        power = powers.get(site, 0)
        ring.add_patch(Circle((x, y), 0.15, zorder=3,
                              facecolor=CALL_FACE if power else "0.95",
                              edgecolor="#8a6d1f" if power else "0.8",
                              lw=2.0 if power else 1.2))
        ring.text(x, y, f"${power}$" if power else "", ha="center", va="center",
                  fontsize=CELL, zorder=4, fontweight="bold", color="0.1")
        ring.text(x * 1.34, y * 1.34, f"$u_{site}$", ha="center", va="center",
                  fontsize=TICK - 1.0, color="0.45")
    ring.plot(np.cos(np.linspace(0, 2 * np.pi, 200)),
              np.sin(np.linspace(0, 2 * np.pi, 200)), color="0.85", lw=1.2, zorder=1)
    ring.annotate("", xy=(np.cos(angles[1]), np.sin(angles[1])),
                  xytext=(np.cos(angles[0]), np.sin(angles[0])), zorder=2,
                  arrowprops={"arrowstyle": "-", "lw": 3.4, "color": "#b3202a"})
    ring.set_xlim(-1.75, 1.75)
    ring.set_ylim(-1.6, 1.5)
    ring.set_aspect("equal")
    ring.axis("off")
    rich_lines(ring, (0.5, 1.10), [f"One shape, {len(family)} kets"],
               TITLE - 0.5, box_alignment=(0.5, 0.0), coords=ring.transAxes)
    fig.tight_layout(w_pad=2.4)
    save(fig, "psc-column")


if __name__ == "__main__":
    print("figures:")
    reach_on_the_lattice()
    shift_symmetry()
    build_trace()
    zone_map()
    zones_split_classes()
    psc_fold()
    psc_elements()
    psc_column()
