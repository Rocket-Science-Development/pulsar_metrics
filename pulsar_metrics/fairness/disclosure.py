"""Disclosure control for group results: complementary (secondary) suppression

Suppressing a small group is not enough when the other numbers let a reader compute it. With one suppressed cell, its
size is the total minus the published sizes of the other groups (lab FA-02, pulsar_metrics 0.15.0); in an
intersection, a suppressed cell is a published margin (e.g. sex=F) minus the other cells of that margin. Rounding the
counts does not help: a group's size can be read back from its exact published rates (k / n).

The rule, from the usual practice of statistical agencies for frequency tables: for the whole feature and for every
margin of an intersection, the suppressed cells number either zero, or at least two holding at least `min_group_size`
units together. Kept cells are suppressed as well, smallest first and never the reference group, until it holds. A
reader can then compute at most a sum of suppressed cells that is itself above the threshold.

Residual risk, documented rather than solved: a margin whose only other cell is the reference group, when the
reference's size can be derived from other published numbers.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def _violations(cells: list, sizes: pd.Series, suppressed: set, min_group_size: int) -> bool:
    hidden = [k for k in cells if k in suppressed]
    return bool(hidden) and (len(hidden) < 2 or float(sizes[hidden].sum()) < min_group_size)


def complementary_suppression(
    sizes: pd.Series, kept: list, suppressed: list, reference, min_group_size: int, depth: int = 1
) -> tuple:
    """(secondary cells, safe): kept groups to suppress as well so that no suppressed cell can be computed by difference

    `sizes` holds the units of each cell, keyed "v1|v2|..." for an intersection of `depth` attributes. The rule is
    applied to the whole feature, then to every margin (cells sharing the values of a proper subset of the attributes),
    until no margin breaks it. `safe` is False when the whole feature still breaks it (only the reference is left): the
    total outside the reference must then not be published."""
    hidden = set(suppressed)
    if not hidden:
        return [], True
    cells = list(sizes.index)
    margins = [()] + [s for d in range(1, depth) for s in combinations(range(depth), d)]
    secondary: list = []
    changed = True
    while changed:
        changed = False
        for subset in margins:
            groups: dict = {}
            for k in cells:
                parts = str(k).split("|")
                groups.setdefault(tuple(parts[i] for i in subset), []).append(k)
            for members in groups.values():
                while _violations(members, sizes, hidden, min_group_size):
                    candidates = sorted(
                        (k for k in members if k not in hidden and k != reference), key=lambda k: (sizes[k], str(k))
                    )
                    if not candidates:
                        break
                    hidden.add(candidates[0])
                    secondary.append(candidates[0])
                    changed = True
    safe = not _violations(cells, sizes, hidden, min_group_size)
    return [k for k in secondary if k in kept], safe
