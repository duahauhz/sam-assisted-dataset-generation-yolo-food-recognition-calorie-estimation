# Coin calibration (cm/pixel)

Source the cm/pixel scale factor from a coin bounding box.

## Algorithm

For each view (top, side):

1. Detect the 1-Yuan coin in the view via YOLO-seg / Faster R-CNN.
2. Take the bounding box of the coin (`[x1, y1, x2, y2]` in pixels).
3. Compute the cm/pixel scale:

    ```
    alpha = 2.5 cm / mean(W, H)
    ```

    where `W = x2 - x1`, `H = y2 - y1`.

4. Use `alpha` as the multiplicative scale factor for that view.

The diameter 2.5 cm comes from the 1-Yuan coin (25 mm diameter, per
`ECUSTFD/paper/1705.07632v3.pdf` §2.2).

## Difference from the MATLAB baseline

The MATLAB runtime code (`ECUSTFD/faster_rcnn/faster_rcnn_rec.m`
line 130) computes:

```matlab
top_pixel = 2.5 / ((y2 + y1 - x2 - x1) / 2);
```

That is `2.5 / (y_sum - x_sum) / 2)`. This expression only equals
`2.5 / (W + H) / 2)` when the bbox is perfectly square. For rotated
or skewed coin detections the two formulas disagree; in practice YOLO
detections on ECUSTFD are within 1-2 %% of square, so the difference
is small.

We use `(W + H) / 2` rather than the MATLAB formula because:

- It is rotation-invariant (a 90-degree rotation does not change the
  answer).
- It does not silently blow up when the bbox is very elongated.
- It reduces to `2.5 / W` when `W = H`, which matches the square-bbox
  limit that the MATLAB formula effectively assumes.

If strict MATLAB bit-for-bit parity is required, see the comment in
`__init__.py` for the one-line swap.

## What it does NOT use

- **No coin mask** is used. The bbox of the detection is the only
  input. The MATLAB baseline behaves the same way. We could refine
  the estimate from the GrabCut mask (equivalent diameter from mask
  pixel count), but this is a deviation from both the paper and the
  MATLAB baseline; document before enabling.

## Files

- `__init__.py` — `compute_coin_scale`, `compute_coin_scales`,
  `filter_coin_detections`, `select_highest_conf`, `CoinScale`.
- `test_coin_scale.py` — unit tests.