"""Unit tests for the calorie-estimation module.

The runtime formula is ``C = q_k * V_tilde``. There is no beta.
"""

from __future__ import annotations

from . import (
    CalorieResultRuntime,
    estimate_calorie_runtime,
    parse_food_info,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
_FOOD_INFO = {
    "apple": {"shape": "ellipsoid", "kcal_per_cm3": 0.4},  # 0.4 kcal/cm^3
    "banana": {"shape": "unknown", "kcal_per_cm3": 0.8},
    "coin": {"shape": None, "kcal_per_cm3": None},
}


def test_runtime_formula_basic() -> None:
    """Apple q=0.4, V=100 -> 40 kcal."""
    r = estimate_calorie_runtime("apple", 100.0, _FOOD_INFO)
    assert isinstance(r, CalorieResultRuntime)
    assert abs(r.kcal - 40.0) < 1e-9
    assert abs(r.q_kcal_per_cm3 - 0.4) < 1e-9


def test_runtime_formula_missing_class() -> None:
    try:
        estimate_calorie_runtime("nonexistent_class", 1.0, _FOOD_INFO)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for missing class")


def test_runtime_formula_coin_no_q() -> None:
    """Coin has q_kcal_per_cm3 = None -> KeyError."""
    try:
        estimate_calorie_runtime("coin", 1.0, _FOOD_INFO)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for coin (no q)")


def test_parse_food_info_real_file() -> None:
    """Parse the real ``food_info.xls`` shipped with the original repo."""
    path = r"E:\AI_Research\dlt8\ECUSTFD\faster_rcnn\food_info.xls"
    info = parse_food_info(path)
    # It is a renamed xlsx file. Verify the structure.
    assert "apple" in info
    assert info["apple"]["shape"] == "ellipsoid"
    assert isinstance(info["apple"]["kcal_per_cm3"], float)
    assert info["apple"]["kcal_per_cm3"] > 0


def run_all() -> None:
    tests = [
        test_runtime_formula_basic,
        test_runtime_formula_missing_class,
        test_runtime_formula_coin_no_q,
        test_parse_food_info_real_file,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
