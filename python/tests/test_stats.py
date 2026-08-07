"""Unit tests for `sceptre_rs_tools.stats` (percentile_r7 and sample-count suppression)."""

from __future__ import annotations

from sceptre_rs_tools.stats import P95_MIN_SAMPLES, P99_MIN_SAMPLES, percentile_r7, suppressed_percentile


def test_percentile_r7_of_empty_list_is_zero() -> None:
    assert percentile_r7([], 0.95) == 0.0


def test_percentile_r7_of_single_value_ignores_fraction() -> None:
    assert percentile_r7([42.0], 0.5) == 42.0
    assert percentile_r7([42.0], 0.99) == 42.0


def test_percentile_r7_median_of_odd_count() -> None:
    assert percentile_r7([1.0, 2.0, 3.0], 0.5) == 2.0


def test_percentile_r7_interpolates_between_ranks() -> None:
    # n=5, p=0.5 -> rank = 0.5*4 = 2.0 -> ordered[2] = 3.0 exactly. ~keep
    assert percentile_r7([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    # n=5, p=0.75 -> rank = 0.75*4 = 3.0 -> ordered[3] = 4.0 exactly. ~keep
    assert percentile_r7([1.0, 2.0, 3.0, 4.0, 5.0], 0.75) == 4.0
    # n=4, p=0.5 -> rank = 0.5*3 = 1.5 -> interpolate ordered[1]=2, ordered[2]=3 -> 2.5. ~keep
    assert percentile_r7([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_percentile_r7_is_unaffected_by_input_order() -> None:
    assert percentile_r7([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == percentile_r7([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)


def test_suppressed_percentile_is_none_below_min_samples() -> None:
    values = [float(i) for i in range(P95_MIN_SAMPLES - 1)]
    assert suppressed_percentile(values, 0.95, P95_MIN_SAMPLES) is None


def test_suppressed_percentile_reports_at_min_samples() -> None:
    values = [float(i) for i in range(P95_MIN_SAMPLES)]
    assert suppressed_percentile(values, 0.95, P95_MIN_SAMPLES) is not None


def test_p99_min_samples_is_larger_than_p95() -> None:
    # p99 needs a deeper sample to mean anything at all. ~keep
    assert P99_MIN_SAMPLES > P95_MIN_SAMPLES
