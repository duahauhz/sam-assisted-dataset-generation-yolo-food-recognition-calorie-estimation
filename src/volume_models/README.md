# Volume models — shape-specific formulas

The five per-shape volume formulas used by both the MATLAB baseline
(`ECUSTFD/faster_rcnn/grabcut_mex.cpp`) and this Python port.

## Provenance

Every formula is a **verbatim port** of `CalVolume()` in
`grabcut_mex.cpp`. The C++ file is the source of truth; the Python
implementation has only been vectorised / split into helper functions
and rewritten in numpy — no formula has been changed.

The ECUSTFD paper itself does **not** list these formulas. §3.4 only
states:

> "we use different formulas to estimate volume of each food.
> After getting volume, food's calorie is obtained by searching
> related tables."

The five formulas below are what the authors chose to implement in
their released code.

## The five formulas

All formulas consume two masks (top view + side view, each cropped to
the food's bounding box) and two scale factors
`alpha_T`, `alpha_S` (cm/pixel, derived from the coin).

Let `sA` = number of foreground pixels in the top mask (or summed
filled widths for column/torus). Let `sB` = sum of squared per-row
foreground pixel counts in the side mask. Let `LB_MAX` = maximum per-row
foreground pixel count in the side mask. Let `alpha_T' = min(alpha_T,
alpha_S * LB_MAX / H_A)` be the cross-view top-scale correction
(present in every shape except ellipsoid).

### 1. ellipsoid

Used for: apple, egg, lemon, orange, peach, plum, qiwi, tomato.

Side view only; top view is ignored. Discretised solid of revolution
(Cavalieri):

```
V = (PI / 4) * alpha_S^3 * sum_i L_i^2
```

where `L_i` = foreground pixel count in row `i` of the side mask.

### 2. column

Used for: bread, sachima.

```
V = (sA * alpha_T'^2) * (mean_height * alpha_S)
```

`sA` = sum of row-span widths in the top mask (filled between first
and last foreground pixel per row).
`mean_height` = mean per-column fill height in the side mask.

### 3. unknown

Used for: banana, bun, fired_dough_twist, litchi, mango, mooncake, pear.

```
V = sA * alpha_T'^2 * sB * alpha_S / LB_MAX^2
```

`sA` = total foreground pixel count in top mask.
`sB` = sum of squared per-row widths in side mask.
`LB_MAX` = max row width in side mask.

### 4. torus

Used for: doughnut.

```
V = (PI^1.5 / 4) * heightB^2 * alpha_T' * alpha_S^2
       * (sqrt(sA + sAE) + sqrt(sAE))
```

`sA` = foreground pixels inside the per-row span in top mask.
`sAE` = non-foreground pixels inside the per-row span (the doughnut
hole).

### 5. grape

Used for: grape.

Same as unknown but each side-row width is multiplied by 0.9 before
squaring (air-gap compensation between grapes in a cluster):

```
V = 0.81 * sA * alpha_T'^2 * sB * alpha_S / LB_MAX^2
```

## Shape-to-class mapping

See `src/constants.py::SHAPE_MODELS`. The mapping is ported from
`grabcut_mex.cpp` lines 9-28 with 4 additional classes inferred
(fired_dough_twist, litchi, mooncake, peach) — see the comments in
that file for details.

## Mask contract

- `top_mask`, `side_mask` are 2-D boolean arrays
  (`True` = foreground).
- `True` corresponds to C++ label `GC_PR_FGD == 3`.
- Masks are cropped to the food bounding box before being passed in
  (see `src/segmentation_runtime/crop_to_bbox.py`).

## Tests

`test_shape_models.py` checks:

- `volume_ellipsoid` matches closed-form `(PI/4) * alpha^3 * w^2 * h`
  for a constant-width side mask.
- `volume_column` reduces to `(H_T * W_T * alpha_T^2) * (H_S * alpha_S)`
  in the limit of large `alpha_S`.
- `volume_grape` equals `0.81 * volume_unknown` for identical masks.
- `volume_torus` returns a positive value and gives a larger volume
  for a ring mask than for a solid disc mask of the same outer
  radius.

## Files

- `shape_models.py` — the five formulas.
- `dispatch.py` — `compute_volume(shape, inputs)` dispatcher.
- `test_shape_models.py` — unit tests.