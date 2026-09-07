"""Pair top-view and side-view detections of the same food class.

The pipeline follows the ECUSTFD paper's matching policy: a top-view
detection is paired with a side-view detection of the *same* class,
and only the highest-confidence detection per class per view is used.
If the class is missing in either view, the pair is skipped — the
paper §4 explicitly drops these "misidentified" images from ME.

Why a separate folder:
- The pairing logic is the bridge between the two-view segmentation
  output and the volume formulas. Getting it wrong silently produces
  wrong-class volumes.
- Easy to unit-test with synthetic detection lists, no images needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from src.coin_calibration import select_highest_conf


@dataclass(frozen=True)
class Detection:
    """Minimal detection record.

    ``mask`` is optional; the volume formulas need it, but the pairing
    logic itself only needs the class name. Keeping mask optional lets
    the pairing be tested without constructing binary masks.
    """

    class_name: str
    conf: float
    bbox: tuple[float, float, float, float]
    mask: object = None  # np.ndarray | None


@dataclass(frozen=True)
class PairedDetection:
    """A top+side pair of the same class, ready for volume computation."""

    class_name: str
    top: Detection
    side: Detection


@dataclass(frozen=True)
class PairingResult:
    """The full pairing outcome for a single image pair.

    ``pairs`` are the classes that appear in BOTH views. ``missing``
    lists the classes that were detected in only one view — useful
    for reporting detection-coverage numbers.
    """

    pairs: List[PairedDetection]
    missing: List[str]


def _per_class_top(detections: Sequence[Detection]) -> dict[str, Detection]:
    """Pick the highest-confidence detection per class."""
    out: dict[str, Detection] = {}
    for d in detections:
        prev = out.get(d.class_name)
        if prev is None or d.conf > prev.conf:
            out[d.class_name] = d
    return out


def _detection_to_dict(d: Detection) -> dict:
    """Normalise a Detection into a dict for the downstream pipeline.

    The volume layer and the orchestrator prefer dicts (so they can
    drop unused fields and re-shape as needed). Keeping the conversion
    here means everything downstream sees a single uniform shape.
    """
    return {
        "class_name": d.class_name,
        "conf": float(d.conf),
        "bbox": tuple(d.bbox),
        "mask": d.mask,
    }


def pair_top_side(
    top_detections: Iterable[Detection],
    side_detections: Iterable[Detection],
    skip_classes: Sequence[str] = ("coin",),
) -> PairingResult:
    """Pair detections of the same class across the two views.

    Args:
        top_detections: detections from the top-view image.
        side_detections: detections from the side-view image.
        skip_classes: classes that should never be paired (e.g. the
            coin, which is used for calibration only and never has a
            volume estimate).

    Returns:
        ``PairingResult`` with the success pairs and the list of
        classes that were detected in only one view.
    """
    top_dets = list(top_detections)
    side_dets = list(side_detections)
    skip = set(skip_classes)

    top_by_cls = {k: v for k, v in _per_class_top(top_dets).items() if k not in skip}
    side_by_cls = {k: v for k, v in _per_class_top(side_dets).items() if k not in skip}

    pairs: List[PairedDetection] = []
    missing: List[str] = []
    for cls in sorted(set(top_by_cls) | set(side_by_cls)):
        if cls in top_by_cls and cls in side_by_cls:
            pairs.append(
                PairedDetection(
                    class_name=cls,
                    top=top_by_cls[cls],
                    side=side_by_cls[cls],
                )
            )
        else:
            missing.append(cls)

    return PairingResult(pairs=pairs, missing=missing)


def pairs_to_dicts(pairs: Sequence[PairedDetection]) -> List[dict]:
    """Convert pairs into a flat list of dicts ready for CSV.

    Each output dict has one row per pair and contains both top and
    side info under ``"top"`` and ``"side"`` keys.
    """
    out: List[dict] = []
    for p in pairs:
        out.append(
            {
                "class_name": p.class_name,
                "top": _detection_to_dict(p.top),
                "side": _detection_to_dict(p.side),
                "top_conf": float(p.top.conf),
                "side_conf": float(p.side.conf),
            }
        )
    return out


def filter_pairs_by_confidence(
    pairs: Sequence[PairedDetection],
    conf_threshold: float,
) -> List[PairedDetection]:
    """Drop pairs whose min(top_conf, side_conf) is below the threshold.

    The paper uses conf >= 0.8 for Faster R-CNN. YOLO-seg tends to
    produce lower confidences, so the pipeline also evaluates at
    conf_threshold in {0.5, 0.8} to compare.
    """
    out: List[PairedDetection] = []
    for p in pairs:
        if min(p.top.conf, p.side.conf) >= conf_threshold:
            out.append(p)
    return out


def filter_pairs_cross_view_soft(
    pairs: Sequence[PairedDetection],
    primary_conf: float = 0.05,
    anchor_conf: Optional[float] = None,
    soft_min_conf: float = 0.01,
    top_1_only: bool = True,
) -> List[PairedDetection]:
    """Confidence-gated matching with top-1 resolution.

    Benchmark protocol (paper-faithful): a pair is kept iff
    ``min(top_conf, side_conf) >= primary_conf``. The gate MUST bind for
    threshold selection to be meaningful — the previous soft gate
    (``c_max >= anchor_conf`` bypass) let any pair with one confident view
    pass at every threshold, making conf a no-op. The anchor bypass is
    still available by passing ``anchor_conf`` explicitly (opt-in).
    If ``top_1_only`` is True, multi-class collisions are resolved by
    keeping the single highest combined score (top_conf * side_conf).
    """
    candidates: List[PairedDetection] = []
    for p in pairs:
        c_min = min(float(p.top.conf), float(p.side.conf))
        keep = c_min >= primary_conf
        if not keep and anchor_conf is not None:
            c_max = max(float(p.top.conf), float(p.side.conf))
            keep = c_max >= anchor_conf and c_min >= soft_min_conf
        if keep:
            candidates.append(p)

    if not top_1_only or len(candidates) <= 1:
        return candidates

    # Disambiguation: Pick the single highest combined score (top_conf * side_conf)
    best_pair = max(candidates, key=lambda p: float(p.top.conf) * float(p.side.conf))
    return [best_pair]


def pair_and_filter_cross_view_soft(
    top_detections: Sequence[Detection],
    side_detections: Sequence[Detection],
    primary_conf: float = 0.05,
    anchor_conf: Optional[float] = None,
    soft_min_conf: float = 0.01,
    top_1_only: bool = True,
    skip_classes: Sequence[str] = ("coin",),
) -> List[PairedDetection]:
    """End-to-end pairing & filtering under a binding confidence gate:

    1. Same-class pairing gated by ``primary_conf`` (hard gate).
    2. Dominant-view fallback for cross-view class mismatches — the rescue
       detections must ALSO satisfy ``primary_conf`` (no rescue with weak
       detections at a high threshold).
    3. No universal low-conf floor: pairs with nothing above the gate
       return [] and are counted as misses in Coverage (never silently
       rescued). Misses are excluded from MAE and reported via Coverage.
    """
    skip = set(skip_classes)
    pairing = pair_top_side(top_detections, side_detections, skip_classes=skip_classes)

    # 1. Attempt same-class matching under the hard gate
    candidates = filter_pairs_cross_view_soft(
        pairing.pairs,
        primary_conf=primary_conf,
        anchor_conf=anchor_conf,
        soft_min_conf=soft_min_conf,
        top_1_only=top_1_only,
    )
    if candidates:
        return candidates

    # 2. Dominant-view fallback for cross-view class mismatches (gate-bound)
    valid_top = [d for d in top_detections if d.class_name not in skip and float(d.conf) >= primary_conf]
    valid_side = [d for d in side_detections if d.class_name not in skip and float(d.conf) >= primary_conf]

    if not valid_top or not valid_side:
        return []

    best_top = max(valid_top, key=lambda d: float(d.conf))
    best_side = max(valid_side, key=lambda d: float(d.conf))

    dominant_class = best_top.class_name if float(best_top.conf) >= float(best_side.conf) else best_side.class_name
    fallback_pair = PairedDetection(
        class_name=dominant_class,
        top=best_top,
        side=best_side,
    )
    return [fallback_pair]


__all__ = [
    "Detection",
    "PairedDetection",
    "PairingResult",
    "pair_top_side",
    "pairs_to_dicts",
    "filter_pairs_by_confidence",
    "filter_pairs_cross_view_soft",
    "pair_and_filter_cross_view_soft",
    "select_highest_conf",  # re-export for convenience
]
