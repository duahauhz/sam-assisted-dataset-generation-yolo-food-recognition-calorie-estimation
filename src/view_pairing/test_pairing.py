"""Unit tests for the view-pairing module."""

from __future__ import annotations

from . import (
    Detection,
    PairingResult,
    filter_pairs_by_confidence,
    pair_top_side,
    pairs_to_dicts,
)


def _det(cls: str, conf: float, bbox=(0, 0, 10, 10)) -> Detection:
    return Detection(class_name=cls, conf=conf, bbox=bbox)


def test_simple_pair() -> None:
    """One apple on both views pairs up."""
    top = [_det("apple", 0.9)]
    side = [_det("apple", 0.8)]
    result = pair_top_side(top, side)
    assert len(result.pairs) == 1
    assert result.pairs[0].class_name == "apple"
    assert result.missing == []


def test_coin_never_paired() -> None:
    """Coin must be skipped even if detected on both views."""
    top = [_det("coin", 0.99)]
    side = [_det("coin", 0.99)]
    result = pair_top_side(top, side)
    assert result.pairs == []
    assert result.missing == []


def test_missing_class_marked() -> None:
    """Apple in top only → 'missing' in result."""
    top = [_det("apple", 0.9)]
    side = [_det("banana", 0.9)]
    result = pair_top_side(top, side)
    assert result.pairs == []
    assert sorted(result.missing) == ["apple", "banana"]


def test_highest_conf_wins() -> None:
    """Two apples on top → pick the higher-conf one."""
    top = [_det("apple", 0.7), _det("apple", 0.95)]
    side = [_det("apple", 0.8)]
    result = pair_top_side(top, side)
    assert len(result.pairs) == 1
    assert result.pairs[0].top.conf == 0.95


def test_multiple_classes() -> None:
    top = [_det("apple", 0.9), _det("banana", 0.85)]
    side = [_det("apple", 0.8), _det("banana", 0.95)]
    result = pair_top_side(top, side)
    assert {p.class_name for p in result.pairs} == {"apple", "banana"}


def test_filter_by_conf() -> None:
    pairs = [
        type("P", (), {})(),
    ]
    # Build a PairedDetection properly
    from . import PairedDetection

    p1 = PairedDetection(
        class_name="apple",
        top=_det("apple", 0.9),
        side=_det("apple", 0.9),
    )
    p2 = PairedDetection(
        class_name="banana",
        top=_det("banana", 0.6),
        side=_det("banana", 0.7),
    )
    kept = filter_pairs_by_confidence([p1, p2], conf_threshold=0.8)
    assert len(kept) == 1
    assert kept[0].class_name == "apple"


def test_pairs_to_dicts() -> None:
    from . import PairedDetection

    p = PairedDetection(
        class_name="apple",
        top=_det("apple", 0.9),
        side=_det("apple", 0.8),
    )
    rows = pairs_to_dicts([p])
    assert len(rows) == 1
    assert rows[0]["class_name"] == "apple"
    assert rows[0]["top_conf"] == 0.9
    assert rows[0]["side_conf"] == 0.8


def run_all() -> None:
    tests = [
        test_simple_pair,
        test_coin_never_paired,
        test_missing_class_marked,
        test_highest_conf_wins,
        test_multiple_classes,
        test_filter_by_conf,
        test_pairs_to_dicts,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
