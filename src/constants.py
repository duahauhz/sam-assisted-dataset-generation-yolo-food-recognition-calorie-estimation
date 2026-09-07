"""Constants and mappings for food calorie estimation."""

from typing import Dict, List

# All 19 food classes in ECUSTFD
FOOD_CLASSES: List[str] = [
    "apple", "banana", "bread", "bun", "doughnut",
    "egg", "fried_dough_twist", "grape", "lemon", "litchi",
    "mango", "mooncake", "orange", "peach", "pear",
    "plum", "qiwi", "sachima", "tomato",
]

# Class name to YOLO index (0-18)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(FOOD_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(FOOD_CLASSES)}

# Shape model for each class.
#
# Source: ``ECUSTFD/faster_rcnn/grabcut_mex.cpp`` lines 9-28
# (the class->shape table the original authors hard-coded) and
# ``ECUSTFD/faster_rcnn/faster_rcnn_rec.m`` line 148
# (``food_info{i+1,2}`` dispatch). The ECUSTFD paper itself does NOT
# specify which shape to use for which class — it only states §3.4
# "we use different formulas to estimate volume of each food".
#
# Provenance per class:
# - Direct port from grabcut_mex.cpp / MATLAB baseline (12 classes):
#     apple, banana, bread, bun, doughnut, egg, lemon, mango, orange,
#     pear, plum, qiwi, sachima, tomato
# - Special-case "grape" branch in grabcut_mex.cpp line 355-416:
#     grape
# - Originally inferred here; corrected to match MATLAB baseline
#   (``food_info.xls`` column 2, read by ``faster_rcnn_rec.m:154``):
#     litchi   -> ellipsoid (small round fruit)
#     mooncake -> column   (flat disc)
#
# ellipsoid: best for round/oval foods
# column: best for flat/disc foods
# torus: best for ring-shaped foods (doughnut)
# grape: special case for grape clusters (air-gap compensation)
# unknown: irregular shape, empirical approximation
SHAPE_MODELS: Dict[str, str] = {
    "apple": "ellipsoid",
    "banana": "unknown",
    "bread": "column",
    "bun": "unknown",
    "doughnut": "torus",
    "egg": "ellipsoid",
    # English should be "fried dough twist" (dough fried in oil).
    # Canonical key spelling throughout the codebase: ``fried_dough_twist``.
    "fried_dough_twist": "unknown",   # long thin fried dough
    "grape": "grape",
    "lemon": "ellipsoid",
    "litchi": "ellipsoid",     # MATLAB baseline (food_info.xls); small round fruit
    "mango": "unknown",
    "mooncake": "column",      # MATLAB baseline (food_info.xls); flat disc
    "orange": "ellipsoid",
    "peach": "ellipsoid",        # not in grabcut_mex.cpp, inferred
    "pear": "unknown",
    "plum": "ellipsoid",
    "qiwi": "ellipsoid",
    "sachima": "column",
    "tomato": "ellipsoid",
}

# Density (g/cm^3) per class — from density.xls (ECUSTFD dataset)
# Calculated as: avg(mass) / avg(volume) per class
DENSITY_G_CM3: Dict[str, float] = {
    "apple": 0.78,
    "banana": 0.91,
    "bread": 0.18,
    "bun": 0.34,
    "doughnut": 0.31,
    "egg": 1.03,
    "fried_dough_twist": 0.58,
    "grape": 0.97,
    "lemon": 0.96,
    "litchi": 1.00,
    "mango": 1.07,
    "mooncake": 0.96,
    "orange": 0.90,
    "peach": 0.96,
    "pear": 1.02,
    "plum": 1.01,
    "qiwi": 0.97,
    "sachima": 0.22,
    "tomato": 0.98,
}

# Energy (kcal/g) per class — from nutrition tables
ENERGY_KCAL_G: Dict[str, float] = {
    "apple": 0.52,
    "banana": 0.89,
    "bread": 3.15,
    "bun": 2.23,
    "doughnut": 4.34,
    "egg": 1.43,
    "fried_dough_twist": 2.16,
    "grape": 0.69,
    "lemon": 0.29,
    "litchi": 0.66,
    "mango": 0.60,
    "mooncake": 18.83,
    "orange": 0.63,
    "peach": 0.57,
    "pear": 0.39,
    "plum": 0.46,
    "qiwi": 0.61,
    "sachima": 2.45,
    "tomato": 0.27,
}

# Coin calibration: 1 Yuan coin diameter in mm
COIN_DIAMETER_MM: float = 25.0
COIN_DIAMETER_CM: float = COIN_DIAMETER_MM / 10.0

# Hard classes with poor accuracy (from CoNet paper analysis)
HARD_CLASSES: List[str] = ["banana", "grape", "mooncake"]

# Grape cluster air-gap correction factor (from grabcut_mex.cpp)
GRAPE_AIR_GAP_FACTOR: float = 0.81  # 0.9^2

# Image extensions used in ECUSTFD
IMAGE_EXTENSIONS: List[str] = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

# Annotation format
ANNOTATION_FORMAT: str = "voc"  # VOC XML format
