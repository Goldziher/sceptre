"""Percentile statistics for the benchmark harness.

``percentile_r7`` is ported from xberg's ``tools/benchmark-harness`` (``src/stats.rs``);
see its docstring for the interpolation method. The sample-count suppression floors below
are new here: sceptre's harness reports p95 from corpus groups as small as 13-27 images,
where a "p95" is just the maximum wearing a statistical label.
"""

from __future__ import annotations

# A percentile computed from too few samples is indistinguishable from the maximum; these
# floors gate whether p95/p99 are reported at all (see `suppressed_percentile`). ~keep
P95_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 100


def percentile_r7(values: list[float], fraction: float) -> float:
    """R-7 (linear interpolation on the empirical CDF) percentile of `values`.

    `fraction` is in `[0, 1]`. This is the same default interpolation method NumPy and R
    use, ported from xberg's `stats.rs::percentile_r7`. An empty list scores `0.0`; a
    single-element list returns that element regardless of `fraction`.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def suppressed_percentile(values: list[float], fraction: float, min_samples: int) -> float | None:
    """`percentile_r7(values, fraction)`, or `None` below `min_samples` samples."""
    if len(values) < min_samples:
        return None
    return percentile_r7(values, fraction)
