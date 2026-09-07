"""Unit tests for the metric helpers."""

from __future__ import annotations

import math

from .metrics import PerClassResult, aggregate, compute_me_per_class


def test_me_perfect_prediction() -> None:
    """v_pred == v_real → ME = 0 for every class."""
    cls = ["apple"] * 5 + ["banana"] * 3
    v_pred = [10.0] * 5 + [20.0] * 3
    v_real = [10.0] * 5 + [20.0] * 3
    out = compute_me_per_class(cls, v_pred, v_real)
    assert len(out) == 2
    for r in out:
        assert math.isclose(r.me_volume_pct, 0.0, abs_tol=1e-9)
        assert r.n_samples == (5 if r.class_name == "apple" else 3)


def test_me_signed_overestimate() -> None:
    """v_pred = 2 * v_real → ME = +100%."""
    out = compute_me_per_class(["apple"], [20.0], [10.0])
    assert math.isclose(out[0].me_volume_pct, 100.0, rel_tol=1e-9)
    assert math.isclose(out[0].abs_me_volume_pct, 100.0, rel_tol=1e-9)


def test_me_signed_underestimate() -> None:
    """v_pred = 0.5 * v_real → ME = -50%."""
    out = compute_me_per_class(["apple"], [5.0], [10.0])
    assert math.isclose(out[0].me_volume_pct, -50.0, rel_tol=1e-9)
    assert math.isclose(out[0].abs_me_volume_pct, 50.0, rel_tol=1e-9)


def test_me_zero_real_skipped() -> None:
    """v_real == 0 → sample is skipped (no division by zero)."""
    out = compute_me_per_class(["apple", "apple"], [10.0, 20.0], [0.0, 10.0])
    assert out[0].n_samples == 1


def test_aggregate() -> None:
    per_class = [
        PerClassResult("apple", 5, 10.0, 10.0, 10.0),
        PerClassResult("banana", 3, -20.0, -20.0, 20.0),
    ]
    overall = aggregate(per_class)
    assert math.isclose(overall["mean_me_volume_pct"], -5.0, rel_tol=1e-9)
    assert math.isclose(overall["mean_abs_me_volume_pct"], 15.0, rel_tol=1e-9)


def test_aggregate_empty() -> None:
    overall = aggregate([])
    assert overall["n_classes"] == 0


def test_me_with_mass() -> None:
    """Mass error is computed when m_pred and m_real are provided."""
    out = compute_me_per_class(
        ["apple"], [10.0], [10.0], m_pred=[15.0], m_real=[10.0]
    )
    assert math.isclose(out[0].me_mass_pct, 50.0, rel_tol=1e-9)


def run_all() -> None:
    tests = [
        test_me_perfect_prediction,
        test_me_signed_overestimate,
        test_me_signed_underestimate,
        test_me_zero_real_skipped,
        test_aggregate,
        test_aggregate_empty,
        test_me_with_mass,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
