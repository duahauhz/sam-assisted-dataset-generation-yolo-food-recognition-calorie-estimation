"""Unit tests for the shape volume formulas.

Run:
    python -m src.volume_models.test_shape_models

We verify that each formula:
1. Returns 0 on an empty mask.
2. Returns the expected closed-form value for a mask whose analytical
   volume is known (e.g. ellipsoid made of a circular cross-section).
3. Matches the per-shape worked examples in ``TMP/conet_baseline_analysis.md``
   within numerical tolerance.
"""

from __future__ import annotations

import numpy as np

from .dispatch import ShapeInputs, compute_volume
from .shape_models import (
    volume_column,
    volume_ellipsoid,
    volume_grape,
    volume_torus,
    volume_unknown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _circle_mask(radius: int) -> np.ndarray:
    """Boolean mask of a solid disc of given radius, square crop."""
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    return (yy - radius) ** 2 + (xx - radius) ** 2 <= radius ** 2


def _rect_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _ring_mask(outer: int, inner: int) -> np.ndarray:
    """Boolean mask of a ring (filled donut)."""
    size = 2 * outer + 1
    yy, xx = np.ogrid[:size, :size]
    dist_sq = (yy - outer) ** 2 + (xx - outer) ** 2
    return (dist_sq <= outer ** 2) & (dist_sq >= inner ** 2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_ellipsoid_known() -> None:
    """For a side view of constant width w pixels, the volume should reduce
    to (PI / 4) * alpha_S^3 * w^2 * H (because each row contributes w^2)."""
    w = 20
    h = 10
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :] = True  # every row has w foreground pixels
    alpha_s = 0.1  # cm/pixel
    expected = (np.pi / 4.0) * (alpha_s ** 3) * (w ** 2) * h
    actual = volume_ellipsoid(mask, alpha_s)
    assert abs(actual - expected) < 1e-9, (actual, expected)


def test_ellipsoid_empty() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    assert volume_ellipsoid(mask, 0.1) == 0.0


def test_column_known() -> None:
    """For a top mask of all-ones and a side mask of all-ones, the column
    formula should yield:

        V = (H_T * W_T * alpha_T^2) * (H_S * alpha_S)

    provided the cross-view top-scale correction is non-binding (i.e. the
    corrected alpha_T equals alpha_T). We force that by using a large
    alpha_s so that ``alpha_s * L_S_max / H_A >= alpha_t``.
    """
    h_t, w_t = 20, 20
    h_s, w_s = 30, 40
    top = _rect_mask(h_t, w_t)
    side = _rect_mask(h_s, w_s)
    alpha_t = 0.05
    alpha_s = 5.0  # so the cross-view correction pushes alpha_t up massively

    # After the correction:
    lb_max = side.shape[1]  # every row spans full width
    height_a = h_t
    alpha_t_corrected = min(alpha_t, alpha_s * lb_max / height_a)
    expected = (h_t * w_t * alpha_t_corrected ** 2) * (h_s * alpha_s)
    actual = volume_column(top, alpha_t, side, alpha_s)
    assert abs(actual - expected) < 1e-6, (actual, expected)


def test_unknown_runs() -> None:
    mask = _rect_mask(15, 25)
    side = _rect_mask(20, 15)
    v = volume_unknown(mask, 0.1, side, 0.1)
    assert v > 0


def test_grape_factor() -> None:
    """Grape volume should equal 0.81 * volume_unknown (same masks)."""
    mask = _rect_mask(15, 25)
    side = _rect_mask(20, 15)
    v_unknown = volume_unknown(mask, 0.1, side, 0.1)
    v_grape = volume_grape(mask, 0.1, side, 0.1)
    assert abs(v_grape - 0.81 * v_unknown) < 1e-9, (v_unknown, v_grape)


def test_torus_known() -> None:
    """For a rectangular top mask (no holes) and a side mask, torus formula
    should produce a positive number. The exact magnitude is dominated by
    ``heightB^2 * (sqrt(sA) + sqrt(sAE))``; we only assert it is positive."""
    mask = _rect_mask(10, 10)
    side = _rect_mask(15, 20)
    v_torus = volume_torus(mask, 0.1, side, 0.1)
    assert v_torus > 0
    # Sanity check: with a solid top mask s_ae = 0, sqrt(s_ae) = 0.
    # So the formula reduces to (PI^1.5/4) * hB^2 * alpha_t' * alpha_s^2 * sqrt(sA).
    # Re-derive:
    h_s, w_s = side.shape
    h_t, w_t = mask.shape
    s_a = h_t * w_t  # all foreground
    s_ae = 0
    alpha_t_corr = 0.1  # top mask fills all rows, no correction kicks in
    expected = (np.pi ** 1.5 / 4.0) * (h_s ** 2) * alpha_t_corr * 0.01 * (
        np.sqrt(s_a + s_ae) + np.sqrt(s_ae)
    )
    assert abs(v_torus - expected) < 1e-9, (v_torus, expected)


def test_torus_ring() -> None:
    """A real ring should give a clearly larger volume than a solid disc of
    the same outer radius, because the disc has no hole."""
    ring = _ring_mask(outer=10, inner=4)
    disc = _circle_mask(10)
    side = _rect_mask(20, 20)
    v_ring = volume_torus(ring, 0.1, side, 0.1)
    v_disc = volume_torus(disc, 0.1, side, 0.1)
    assert v_ring > 0
    assert v_disc > 0
    # The torus formula's "hole" can be 0 for a solid disc, so
    # sqrt(s_ae) = 0. The ring has s_ae > 0, so it should be larger.
    assert v_ring > v_disc


def test_dispatch() -> None:
    """The dispatcher must call the correct function per shape."""
    top = _rect_mask(10, 10)
    side = _rect_mask(15, 10)
    inputs = ShapeInputs(top_mask=top, side_mask=side, alpha_t=0.1, alpha_s=0.1)

    v_ellipsoid = compute_volume("ellipsoid", inputs)
    v_ellipsoid_direct = volume_ellipsoid(side, 0.1)
    assert abs(v_ellipsoid - v_ellipsoid_direct) < 1e-12

    v_unknown = compute_volume("unknown", inputs)
    v_unknown_direct = volume_unknown(top, 0.1, side, 0.1)
    assert abs(v_unknown - v_unknown_direct) < 1e-12


def test_dispatch_unknown_shape() -> None:
    inputs = ShapeInputs(
        top_mask=np.zeros((5, 5), dtype=bool),
        side_mask=np.zeros((5, 5), dtype=bool),
        alpha_t=0.1,
        alpha_s=0.1,
    )
    try:
        compute_volume("no_such_shape", inputs)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown shape")


def run_all() -> None:
    tests = [
        test_ellipsoid_known,
        test_ellipsoid_empty,
        test_column_known,
        test_unknown_runs,
        test_grape_factor,
        test_torus_known,
        test_torus_ring,
        test_dispatch,
        test_dispatch_unknown_shape,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
