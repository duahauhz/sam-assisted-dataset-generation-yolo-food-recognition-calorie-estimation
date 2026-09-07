"""Unit tests for the pairing helpers in dataset_split."""

from __future__ import annotations

from pathlib import Path

from .dataset_split import (
    group_top_side,
    make_pairs,
    parse_filename,
    resolve_image_paths,
)


def test_parse_filename() -> None:
    p = parse_filename("apple001S(1).JPG")
    assert p == ("apple001", "S", 1)
    p = parse_filename("apple001T(2).jpg")
    assert p == ("apple001", "T", 2)
    assert parse_filename("foo.txt") is None


def test_group_top_side() -> None:
    paths = [
        Path("apple001T(1).JPG"),
        Path("apple001S(1).JPG"),
        Path("apple001S(2).JPG"),
        Path("apple002T(1).JPG"),
    ]
    groups = group_top_side(paths)
    assert len(groups) == 2
    apple001 = next(
        g for g in groups
        if any(p.name == "apple001T(1).JPG" for p in g["top"])
    )
    assert len(apple001["top"]) == 1
    assert len(apple001["side"]) == 2


def test_make_pairs_cartesian() -> None:
    groups = [
        {"top": [Path("apple001T(1).JPG")], "side": [Path("apple001S(1).JPG"), Path("apple001S(2).JPG")]},
    ]
    pairs = make_pairs(groups)
    assert len(pairs) == 2


def test_resolve_image_paths() -> None:
    """Resolver tries multiple extensions and skips missing files."""
    images_dir = Path("E:/AI_Research/dlt8/data/raw/ECUSTFD/JPEGImages")
    paths = resolve_image_paths(["apple001T(1)", "nonexistent"], images_dir)
    assert len(paths) == 1
    assert paths[0].name.upper().startswith("APPLE001T")


def run_all() -> None:
    tests = [
        test_parse_filename,
        test_group_top_side,
        test_make_pairs_cartesian,
        test_resolve_image_paths,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
