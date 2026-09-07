"""Shape-specific volume formulas ported from ECUSTFD/grabcut_mex.cpp.

Each function takes the foreground mask (boolean numpy array, where
``True`` represents pixels belonging to the food portion — equivalent to
the C++ label ``GC_PR_FGD == 3``) and the per-view scale ``alpha`` in
cm/pixel.

Sign convention (matches C++):
- ``top_mask`` shape (H_T, W_T) — foreground pixels registered to the
  top-down view crop.
- ``side_mask`` shape (H_S, W_S) — foreground pixels registered to the
  side view crop.
- Row index = y (height), column index = x (width).

All formulas are verbatim ports; only cosmetic refactoring (vectorised
numpy, function decomposition) has been applied. The masked pixel
identity is intentionally simple: ``mask == 3`` in C++ becomes
``mask == True`` here.
"""

from __future__ import annotations

import numpy as np

# C++ uses two precision constants; we keep both for clarity.
_PI = 3.14159265358979323846
_GRAPE_AIR_GAP = 0.9  # 0.9^2 = 0.81 — the per-row scaling for grape clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_foreground_counts(mask: np.ndarray) -> np.ndarray:
    """Return number of foreground pixels per row.

    For ``shape == "column"`` / ``"torus"`` the C++ implementation uses
    fills *between* the first and last foreground pixel of each row; we
    expose that as a separate helper below. Functions that count raw
    foreground pixels use this helper.
    """
    return mask.sum(axis=1).astype(np.int64)


def _row_filled_widths(mask: np.ndarray) -> np.ndarray:
    """Return the span width between first and last foreground pixel of each row.

    Rows with no foreground return 0. Used by the column and torus
    formulas, which collapse holes inside the contour (GrabCut post-
    processing tends to leave occasional interior background pixels).
    """
    h, w = mask.shape
    widths = np.zeros(h, dtype=np.int64)
    has_any = mask.any(axis=1)
    # argmax finds the first True; for last True we flip horizontally.
    first = np.argmax(mask, axis=1)
    last = w - 1 - np.argmax(mask[:, ::-1], axis=1)
    widths[has_any] = last[has_any] - first[has_any] + 1
    return widths


def _row_filled_widths_with_height(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Like _row_filled_widths but also returns the effective heightA.

    Rows with no foreground decrement heightA (matches C++ lines
    228-232 / 442-446). heightA never goes below 1 to avoid division by
    zero in the scale correction.
    """
    h, w = mask.shape
    has_any = mask.any(axis=1)
    height = int(has_any.sum())
    if height == 0:
        height = 1
    first = np.argmax(mask, axis=1)
    last = w - 1 - np.argmax(mask[:, ::-1], axis=1)
    widths = np.zeros(h, dtype=np.int64)
    widths[has_any] = last[has_any] - first[has_any] + 1
    return widths, height


def _column_filled_heights(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (column heights, valid-column count) used by ``column``.

    Valid columns are those containing at least one foreground pixel.
    """
    h, w = mask.shape
    has_any = mask.any(axis=0)
    valid = int(has_any.sum())
    if valid == 0:
        valid = 1
    first = np.argmax(mask, axis=0)
    last = h - 1 - np.argmax(mask[::-1, :], axis=0)
    heights = np.zeros(w, dtype=np.int64)
    heights[has_any] = last[has_any] - first[has_any] + 1
    return heights, valid


# ---------------------------------------------------------------------------
# Core shape formulas (verbatim ports)
# ---------------------------------------------------------------------------
def volume_ellipsoid(side_mask: np.ndarray, alpha_s: float) -> float:
    """Ellipsoid formula — uses side view only.

    V = (PI / 4) * alpha_S^3 * Σ_i L_i^2

    The top view is intentionally ignored (per paper §3.5 and the C++
    implementation). The integral of the squared side-row width is the
    discretised solid of revolution around the vertical axis.
    """
    if side_mask.ndim != 2:
        raise ValueError("side_mask must be 2-D (H, W)")
    rows = _row_foreground_counts(side_mask)
    return float(_PI / 4.0 * (alpha_s ** 3) * np.sum(rows.astype(np.float64) ** 2))


def volume_column(
    top_mask: np.ndarray, alpha_t: float, side_mask: np.ndarray, alpha_s: float
) -> float:
    """Column formula — uses top view (filled area) and side view (mean height).

    V = (sA * alpha_T'^2) * (mean_height / rows * alpha_S)

    where ``alpha_T' = min(alpha_T, alpha_S * L_S_max / H_A)`` is the
    cross-view top-scale correction (C++ line 277).
    """
    if top_mask.ndim != 2 or side_mask.ndim != 2:
        raise ValueError("top_mask and side_mask must be 2-D (H, W)")

    # Top: fill between first and last foreground pixel of each row.
    widths, height_a = _row_filled_widths_with_height(top_mask)
    s_a = int(widths.sum())

    # Side: per-row span (LB_MAX) and per-column fill height.
    side_widths, _ = _row_filled_widths_with_height(side_mask)
    lb_max = int(side_widths.max()) if side_widths.size else 0

    side_heights, valid_cols = _column_filled_heights(side_mask)
    mean_height = float(side_heights.sum()) / float(valid_cols)

    alpha_t_corrected = min(alpha_t, alpha_s * lb_max / height_a)
    return float(s_a * (alpha_t_corrected ** 2) * (mean_height * alpha_s))


def volume_unknown(
    top_mask: np.ndarray, alpha_t: float, side_mask: np.ndarray, alpha_s: float
) -> float:
    """Unknown (irregular) shape — weighted solid-of-revolution.

    V = sA * alpha_T'^2 * sB * alpha_S / L_S_max^2

    where ``sA = sum of foreground pixels in top view``,
    ``sB = sum of L_i^2`` (L_i = foreground pixels in side row i), and
    ``alpha_T' = min(alpha_T, alpha_S * L_S_max / H_A)``.
    """
    if top_mask.ndim != 2 or side_mask.ndim != 2:
        raise ValueError("top_mask and side_mask must be 2-D (H, W)")

    s_a = int(top_mask.sum())
    if s_a == 0:
        return 0.0

    # The heightA variable is tracked but unused in the C++ code for
    # "unknown" (the row-without-foreground decrement is commented out).
    # We mirror that behaviour: heightA is constant top.rows.
    height_a = top_mask.shape[0]

    side_rows = _row_foreground_counts(side_mask)
    s_b = float(np.sum(side_rows.astype(np.float64) ** 2))
    lb_max = int(side_rows.max()) if side_rows.size else 0
    if lb_max == 0:
        return 0.0

    alpha_t_corrected = min(alpha_t, alpha_s * lb_max / height_a)
    return float(s_a * (alpha_t_corrected ** 2) * s_b * alpha_s / (lb_max ** 2))


def volume_grape(
    top_mask: np.ndarray, alpha_t: float, side_mask: np.ndarray, alpha_s: float
) -> float:
    """Grape cluster formula — same as ``unknown`` but each side row width is
    multiplied by 0.9 before squaring (air-gap compensation, C++ line 394).

    V = sA * alpha_T'^2 * Σ (0.9 * L_i)^2 * alpha_S / L_S_max^2
        = 0.81 * sA * alpha_T'^2 * sB * alpha_S / L_S_max^2
    """
    if top_mask.ndim != 2 or side_mask.ndim != 2:
        raise ValueError("top_mask and side_mask must be 2-D (H, W)")

    s_a = int(top_mask.sum())
    if s_a == 0:
        return 0.0

    height_a = top_mask.shape[0]
    side_rows = _row_foreground_counts(side_mask)
    s_b = float(np.sum((side_rows.astype(np.float64) * _GRAPE_AIR_GAP) ** 2))
    lb_max = int(side_rows.max()) if side_rows.size else 0
    if lb_max == 0:
        return 0.0

    alpha_t_corrected = min(alpha_t, alpha_s * lb_max / height_a)
    return float(s_a * (alpha_t_corrected ** 2) * s_b * alpha_s / (lb_max ** 2))


def volume_torus(
    top_mask: np.ndarray, alpha_t: float, side_mask: np.ndarray, alpha_s: float
) -> float:
    """Torus formula (doughnut-shaped) — uses both views.

    V = (PI^1.5 / 4) * heightB^2 * alpha_T' * alpha_S^2
        * (sqrt(sA + sAE) + sqrt(sAE))

    where ``sA`` counts foreground pixels in the top view, ``sAE``
    counts non-foreground pixels *inside* the row span (the doughnut
    hole), and ``alpha_T' = min(alpha_T, alpha_S * L_S_max / H_A)``.
    Note that the C++ source uses ``heightB`` after the row-without-
    foreground decrement (lines 449-466), not the raw row count.
    """
    if top_mask.ndim != 2 or side_mask.ndim != 2:
        raise ValueError("top_mask and side_mask must be 2-D (H, W)")

    # Top: split pixels inside the row span into foreground vs hole.
    h, w = top_mask.shape
    has_any = top_mask.any(axis=1)
    height_a = int(has_any.sum())
    if height_a == 0:
        height_a = 1

    first = np.argmax(top_mask, axis=1)
    last = w - 1 - np.argmax(top_mask[:, ::-1], axis=1)

    s_a = 0
    s_ae = 0
    for i in np.flatnonzero(has_any):
        span = top_mask[i, first[i]:last[i] + 1]
        s_a += int(span.sum())
        s_ae += int(span.size - span.sum())

    if s_a <= 0:
        return 0.0

    # Side: only L_S_max is used, computed on rows-with-foreground
    # basis; heightB is the number of valid rows.
    side_widths, height_b = _row_filled_widths_with_height(side_mask)
    if side_widths.size == 0 or height_b == 0:
        return 0.0
    lb_max = int(side_widths.max())

    alpha_t_corrected = min(alpha_t, alpha_s * lb_max / height_a)
    return float(
        (_PI ** 1.5)
        * (height_b ** 2)
        * alpha_t_corrected
        * (alpha_s ** 2)
        * (np.sqrt(s_a + s_ae) + np.sqrt(s_ae))
        / 4.0
    )
