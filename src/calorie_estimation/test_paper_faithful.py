"""Unit tests for paper-faithful calorie estimation.

Paper Table 1 (Liang & Li 2017, page 2) defines:
- Density (g/cm^3) per class
- Energy (kcal/g) per class

Module formula: ``q_paper = rho * energy``, ``C = q * V``.

The tests verify:
1. q-values derived from Table 1 match what the paper would compute.
2. Cross-check: q-paper matches ``food_info.xls`` (runtime MATLAB) within
   ~5% for all 17 "agreeing" classes.
3. Calorie = q * V.
"""

from __future__ import annotations

from . import (
    PAPER_DENSITY_G_CM3,
    PAPER_ENERGY_PER_G,
    paper_calorie_kcal,
    paper_q_kcal_per_cm3,
    parse_food_info,
)


def test_paper_density_count() -> None:
    """Table 1 has 19 foods (excluding 'mix')."""
    assert len(PAPER_DENSITY_G_CM3) == 19
    assert len(PAPER_ENERGY_PER_G) == 19


def test_paper_q_apple() -> None:
    """apple: rho=0.78, energy=0.52 → q=0.4056."""
    q = paper_q_kcal_per_cm3("apple")
    assert q is not None
    assert abs(q - 0.4056) < 1e-4


def test_paper_q_mooncake() -> None:
    """mooncake: rho=0.96, energy=18.83 → q=18.0768."""
    q = paper_q_kcal_per_cm3("mooncake")
    assert q is not None
    assert abs(q - 18.0768) < 1e-4


def test_paper_q_unknown() -> None:
    """Unknown class → None."""
    assert paper_q_kcal_per_cm3("not_a_real_class") is None


def test_paper_calorie_apple() -> None:
    """apple: q=0.4056, V=100 → 40.56."""
    c = paper_calorie_kcal("apple", 100.0)
    assert c is not None
    assert abs(c - 40.56) < 1e-3


def test_paper_calorie_unknown() -> None:
    """Unknown class → None."""
    assert paper_calorie_kcal("fake_food", 100.0) is None


def test_paper_vs_xls_consistency() -> None:
    """q-paper should match q-xls within 5% for the 'agreeing' classes.

    Outliers (peach, pear) are excluded because food_info.xls does NOT
    match Table 1 for them (peach ratio 1.37, pear ratio 0.73 in our
    earlier audit). Those are xls-side quirks, not paper-side issues.
    """
    info = parse_food_info(
        r"E:\AI_Research\dlt8\data\raw\ECUSTFD\paper\food_info.xls"
    )
    excluded = {"peach", "pear"}  # xls-side quirks
    for cls in PAPER_DENSITY_G_CM3:
        if cls in excluded:
            continue
        q_paper = paper_q_kcal_per_cm3(cls)
        q_xls = info.get(cls, {}).get("kcal_per_cm3")
        if q_xls is None or q_paper is None:
            continue
        ratio = q_paper / q_xls
        assert 0.95 < ratio < 1.05, (
            f"{cls}: paper={q_paper:.4f} xls={q_xls:.4f} ratio={ratio:.4f}"
        )


def run_all() -> None:
    tests = [
        test_paper_density_count,
        test_paper_q_apple,
        test_paper_q_mooncake,
        test_paper_q_unknown,
        test_paper_calorie_apple,
        test_paper_calorie_unknown,
        test_paper_vs_xls_consistency,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()