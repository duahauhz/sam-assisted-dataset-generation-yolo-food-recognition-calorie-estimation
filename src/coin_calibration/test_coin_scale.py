"""Unit tests for the coin-calibration module."""

from __future__ import annotations

import math

from src.constants import COIN_DIAMETER_MM

from . import (
    COIN_DIAMETER_CM,
    compute_coin_scale,
    compute_coin_scales,
    filter_coin_detections,
    select_highest_conf,
)


def test_diameter_constant() -> None:
    """The cm constant must be exactly 1/10 of the mm constant."""
    assert COIN_DIAMETER_CM == COIN_DIAMETER_MM / 10.0
    assert abs(COIN_DIAMETER_CM - 2.5) < 1e-9


def test_perfect_square() -> None:
    """A perfectly square 100x100 px coin bbox gives alpha = 2.5/100 = 0.025."""
    scale = compute_coin_scale([0, 0, 100, 100])
    assert math.isclose(scale.alpha_cm_per_pixel, 0.025, rel_tol=1e-9)


def test_rectangular() -> None:
    """A 120x80 bbox gives alpha = 2.5 / 100 = 0.025."""
    scale = compute_coin_scale([0, 0, 120, 80])
    assert math.isclose(scale.alpha_cm_per_pixel, 0.025, rel_tol=1e-9)


def test_degenerate_bbox() -> None:
    try:
        compute_coin_scale([10, 10, 10, 10])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for zero-size bbox")


def test_scales_dict() -> None:
    """``compute_coin_scales`` should handle Nones correctly."""
    out = compute_coin_scales({"top": [0, 0, 100, 100], "side": None})
    assert out["top"] is not None and out["top"].alpha_cm_per_pixel == 0.025
    assert out["side"] is None


def test_filter_coins() -> None:
    dets = [
        {"class_name": "coin", "conf": 0.9},
        {"class_name": "apple", "conf": 0.8},
        {"class_name": "coin", "conf": 0.7},
    ]
    coins = filter_coin_detections(dets)
    assert len(coins) == 2
    assert all(d["class_name"] == "coin" for d in coins)


def test_select_highest_conf() -> None:
    dets = [
        {"class_name": "coin", "conf": 0.6},
        {"class_name": "coin", "conf": 0.9},
        {"class_name": "coin", "conf": 0.7},
    ]
    best = select_highest_conf(dets)
    assert best is not None and best["conf"] == 0.9

    assert select_highest_conf([]) is None


def run_all() -> None:
    tests = [
        test_diameter_constant,
        test_perfect_square,
        test_rectangular,
        test_degenerate_bbox,
        test_scales_dict,
        test_filter_coins,
        test_select_highest_conf,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
