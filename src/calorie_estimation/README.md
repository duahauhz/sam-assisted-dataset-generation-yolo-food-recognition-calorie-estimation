# Calorie estimation

Convert a raw geometric volume (`V_tilde` in cm^3) into calories.

## The single formula

```
C = q_k * V_tilde       (kcal)
```

where `q_k = rho_k * c_k` is the per-class kcal/cm^3 factor
pre-computed and stored in `food_info.xls` (a renamed xlsx file
shipped with `ECUSTFD/faster_rcnn/`). This is exactly what the
runtime MATLAB code does (`ECUSTFD/faster_rcnn/faster_rcnn_rec.m`
line 156):

```matlab
calorie(i) = volume * food_info{i+1,3};
```

## Provenance

- The ECUSTFD paper only states (§3.4): "After getting volume,
  food's calorie is obtained by searching related tables." It does
  not give an explicit formula.
- The runtime MATLAB code uses `q_k * V_tilde`. This is the
  canonical implementation.
- The paper does **not** contain a beta correction. Beta appears
  only in the offline analysis script
  `ECUSTFD/faster_rcnn/xls_results_analysis.m` (lines 96-97, 134-136)
  and is **not** a runtime formula. It has been removed from this
  codebase.

## Density and energy tables

`q_k` is read from `food_info.xls` via the parser at
`food_info_xls.py`. The same `rho` and `c` are also duplicated in
`src/constants.py` (`DENSITY_G_CM3` and `ENERGY_KCAL_G`) for the
evaluation pipeline.

## Files

- `food_info_xls.py` — parser for the original `food_info.xls`.
- `runtime_formula.py` — `estimate_calorie_runtime` (the only
  calorie estimator in the codebase).
- `__init__.py` — re-exports.
- `test_calorie.py` — unit tests.