# mlrunstats/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Summary:
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    cv: float


def _as_clean_array(x: List[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    return a


def summarize(
    x: List[float],
    alpha: float = 0.05,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Summary:
    """Mean, std (ddof=1), CV; 95% CI via bootstrap on the mean."""
    a = _as_clean_array(x)
    n = a.size
    if n == 0:
        return Summary(np.nan, np.nan, 0, np.nan, np.nan, np.nan)

    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    cv = (std / mean) if mean != 0 else np.inf

    if n == 1:
        # CI collapses to the single value
        return Summary(mean, std, n, mean, mean, cv)

    if rng is None:
        rng = np.random.default_rng()

    # Bootstrap means
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = a[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return Summary(mean, std, n, lo, hi, cv)


def one_sample_greater(
    x: List[float],
    mu0: float,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    One-sample bootstrap p-value for H1: mean(x) > mu0.
    p ≈ fraction of bootstrap means ≤ mu0 (smoothed).
    """
    a = _as_clean_array(x)
    n = a.size
    if n == 0:
        return np.nan

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = a[idx].mean(axis=1)

    # One-sided (greater): small p when boot_means mostly > mu0.
    # Add +1 smoothing to avoid zero p-values.
    num = int(np.sum(boot_means <= mu0)) + 1
    den = n_boot + 1
    return num / den


def two_sample_greater(
    x: List[float],
    y: List[float],
    n_perm: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Permutation test on difference of means for H1: mean(x) > mean(y).
    Returns one-sided p-value: P(diff_perm >= diff_obs | H0).
    """
    a = _as_clean_array(x)
    b = _as_clean_array(y)
    nx, ny = a.size, b.size
    if nx == 0 or ny == 0:
        return np.nan

    if rng is None:
        rng = np.random.default_rng()

    diff_obs = float(a.mean() - b.mean())

    pool = np.concatenate([a, b], axis=0)
    n = pool.size
    # Use label permutation
    count = 1  # smoothing
    for _ in range(n_perm):
        perm = rng.permutation(n)
        x_idx = perm[:nx]
        y_idx = perm[nx:]
        diff = float(pool[x_idx].mean() - pool[y_idx].mean())
        if diff >= diff_obs:
            count += 1
    return count / (n_perm + 1)


def effect_sizes(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Cohen's d (pooled SD), Hedges' g (small-sample correction), Cliff's delta.
    """
    a = _as_clean_array(x)
    b = _as_clean_array(y)
    nx, ny = a.size, b.size

    # Handle edge cases
    if nx == 0 or ny == 0:
        return {"cohens_d": np.nan, "hedges_g": np.nan, "cliffs_delta": np.nan}

    mx, my = float(a.mean()), float(b.mean())
    vx = float(a.var(ddof=1)) if nx > 1 else 0.0
    vy = float(b.var(ddof=1)) if ny > 1 else 0.0

    # Pooled SD
    denom_df = (nx + ny - 2)
    if denom_df > 0:
        sp2 = ((nx - 1) * vx + (ny - 1) * vy) / denom_df
        sp = np.sqrt(sp2)
    else:
        sp = 0.0

    if sp > 0:
        d = (mx - my) / sp
    else:
        # If pooled SD is 0, define d as inf if means differ, else 0
        d = np.inf if (mx != my) else 0.0

    # Hedges' g correction
    # J ≈ 1 - 3/(4*(nx+ny) - 9)
    n_tot = nx + ny
    if n_tot > 2:
        J = 1.0 - 3.0 / (4.0 * n_tot - 9.0)
    else:
        J = 1.0
    g = d * J

    # Cliff's delta: P(a > b) - P(a < b)
    # For n≈40, brute-force pairs is fine.
    # Efficient version possible, but not necessary here.
    greater = 0
    less = 0
    for va in a:
        greater += int(np.sum(va > b))
        less += int(np.sum(va < b))
    cliffs = (greater - less) / float(nx * ny)

    return {"cohens_d": float(d), "hedges_g": float(g), "cliffs_delta": float(cliffs)}
