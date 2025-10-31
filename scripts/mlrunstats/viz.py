# mlrunstats/viz.py
from __future__ import annotations

from typing import List, Sequence

# Try importing matplotlib. If not available, define no-op stubs.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def hist(values: Sequence[float], title: str, out_png: str, bins: int = 15) -> None:
    """Save a histogram of values. One figure per call."""
    if not _HAS_MPL:
        return
    import numpy as np

    a = [v for v in values if v == v]  # drop NaN
    if len(a) == 0:
        return

    plt.figure()
    plt.hist(a, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def box(values_a: Sequence[float], values_b: Sequence[float],
        labels: List[str], title: str, out_png: str) -> None:
    """Boxplot comparing two sets of values. One figure per call."""
    if not _HAS_MPL:
        return

    data = []
    for vals in (values_a, values_b):
        data.append([v for v in vals if v == v])

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def violin(values_a: Sequence[float], values_b: Sequence[float],
           labels: List[str], title: str, out_png: str) -> None:
    """Violin plot comparing two sets of values. One figure per call."""
    if not _HAS_MPL:
        return

    data = []
    for vals in (values_a, values_b):
        data.append([v for v in vals if v == v])

    plt.figure()
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title(title)
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
