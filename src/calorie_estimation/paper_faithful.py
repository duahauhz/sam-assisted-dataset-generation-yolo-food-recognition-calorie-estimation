"""Paper-faithful calorie estimation (Liang & Li 2017, arXiv:1705.07632v3).

This module reproduces the calorie-estimation convention used in the
ECUSTFD paper, Table 1 (page 2). It is NOT a correction or
normalization — it is an exact replica of what the authors put in the
paper, so the codebase can faithfully mimic their pipeline.

The runtime MATLAB baseline (``ECUSTFD/faster_rcnn/faster_rcnn_rec.m``
line 156) computes:

    calorie(i) = volume * food_info{i+1, 3};

where column 3 of ``food_info.xls`` is the per-class q-factor in
kcal/cm^3. The xls file in turn stores ``q = rho * energy`` for each
class, with ``rho`` and ``energy`` taken verbatim from Table 1 of the
paper.

Paper Table 1 (verbatim — Density in g/cm^3, Energy labeled "kcal/g"):

    | class              | rho  | energy  | q = rho * energy |
    | ------------------ | ---- | ------- | ---------------- |
    | apple              | 0.78 | 0.52    | 0.4056           |
    | banana             | 0.91 | 0.89    | 0.8099           |
    | bread              | 0.18 | 3.15    | 0.5670           |
    | bun                | 0.34 | 2.23    | 0.7582           |
    | doughnut           | 0.31 | 4.34    | 1.3454           |
    | egg                | 1.03 | 1.43    | 1.4729           |
    | fried dough twist  | 0.58 | 24.16   | 14.0128          |
    | grape              | 0.97 | 0.69    | 0.6693           |
    | lemon              | 0.96 | 0.29    | 0.2784           |
    | litchi             | 1.00 | 0.66    | 0.6600           |
    | mango              | 1.07 | 0.60    | 0.6420           |
    | mooncake           | 0.96 | 18.83   | 18.0768          |
    | orange             | 0.90 | 0.63    | 0.5670           |
    | peach              | 0.96 | 0.57    | 0.5472           |
    | pear               | 1.02 | 0.39    | 0.3978           |
    | plum               | 1.01 | 0.46    | 0.4646           |
    | qiwi               | 0.97 | 0.61    | 0.5917           |
    | sachima            | 0.22 | 21.45   | 4.7190           |
    | tomato             | 0.98 | 0.27    | 0.2646           |

References (verbatim quotes):
- "After getting volume, food's calorie is obtained by searching related
  tables."  — Liang & Li 2017, §3.4 page 4.
- "The density is calculated with the volume and mass information
  collected in ECUSTFD. For each kind of food, energy is obtained from
  nutrition table."  — Liang & Li 2017, §2.1 page 2.
- Table 1 ("Density (g/cm3)" and "Energy (kcal/g)" columns), page 2.

Known caveats (do NOT silently fix — paper-faithful = paper-bug too):
1. Three asian foods (mooncake, sachima, fried dough twist) have
   "Energy (kcal/g)" values that, when divided by 4.184, match USDA
   kcal/100g (~1.07-1.09 ratio). This strongly suggests the authors
   took these values from a Chinese food-composition table that
   reports kJ/g and the unit label was inadvertently kept as "kcal/g".
2. Other classes use values from a Chinese food table that differ
   from USDA (lower per 100g than USDA) but are internally consistent
   with kcal/g semantics.

Use this module when you want to reproduce paper numbers exactly.
Use ``runtime_formula.py`` + ``food_info.xls`` when you want the
xls-derived q values (which equal ``rho * energy`` within ~4% for 17
of 19 classes).
"""

from __future__ import annotations

from typing import Dict, Optional


# Verbatim from Table 1 of Liang & Li 2017, arXiv:1705.07632v3 page 2.
# Density column header: "Density (g/cm3)"
PAPER_DENSITY_G_CM3: Dict[str, float] = {
    "apple":             0.78,
    "banana":            0.91,
    "bread":             0.18,
    "bun":               0.34,
    "doughnut":          0.31,
    "egg":               1.03,
    "fried_dough_twist": 0.58,
    "grape":             0.97,
    "lemon":             0.96,
    "litchi":            1.00,
    "mango":             1.07,
    "mooncake":          0.96,
    "orange":            0.90,
    "peach":             0.96,
    "pear":              1.02,
    "plum":              1.01,
    "qiwi":              0.97,
    "sachima":           0.22,
    "tomato":            0.98,
}

# Verbatim from Table 1 of Liang & Li 2017, arXiv:1705.07632v3 page 2.
# Column header: "Energy (kcal/g)" — note: see module docstring caveat.
PAPER_ENERGY_PER_G: Dict[str, float] = {
    "apple":             0.52,
    "banana":            0.89,
    "bread":             3.15,
    "bun":               2.23,
    "doughnut":          4.34,
    "egg":               1.43,
    "fried_dough_twist": 24.16,
    "grape":             0.69,
    "lemon":             0.29,
    "litchi":            0.66,
    "mango":             0.60,
    "mooncake":          18.83,
    "orange":            0.63,
    "peach":             0.57,
    "pear":              0.39,
    "plum":              0.46,
    "qiwi":              0.61,
    "sachima":           21.45,
    "tomato":            0.27,
}


def paper_q_kcal_per_cm3(class_name: str) -> Optional[float]:
    """Per-class q-factor derived from paper Table 1: ``q = rho * energy``.

    Returns the q-factor in the unit produced by ``rho [g/cm^3] *
    energy [per the paper's 'kcal/g' label]`` = whatever the paper's
    Table 1 says (NOT normalised to true kcal/cm^3 — see module
    docstring caveat). Use the same convention downstream if you want
    paper-faithful numbers.

    Args:
        class_name: e.g. ``"apple"`` (uses snake_case keys).

    Returns:
        q in paper's convention, or ``None`` if the class is unknown.
    """
    rho = PAPER_DENSITY_G_CM3.get(class_name)
    e = PAPER_ENERGY_PER_G.get(class_name)
    if rho is None or e is None:
        return None
    return float(rho) * float(e)


def paper_calorie_kcal(class_name: str, v_cm3: float) -> Optional[float]:
    """Paper-faithful calorie estimate: ``C = q_paper * V``.

    Args:
        class_name: e.g. ``"apple"``.
        v_cm3: raw geometric volume (cm^3) from shape model.

    Returns:
        Calorie value in the paper's convention (same caveat as
        ``paper_q_kcal_per_cm3``). Returns ``None`` for unknown class.
    """
    q = paper_q_kcal_per_cm3(class_name)
    if q is None:
        return None
    return float(q) * float(v_cm3)


__all__ = [
    "PAPER_DENSITY_G_CM3",
    "PAPER_ENERGY_PER_G",
    "paper_q_kcal_per_cm3",
    "paper_calorie_kcal",
]