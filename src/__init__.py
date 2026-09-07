"""Food Volume & Calorie Estimation from Single RGB Image.

Module layout:
    data_prep    — Dataset loading, SAM mask generation, VOC→YOLO conversion
    geometry     — Homography, calibration, shape models
    detection    — YOLO detection wrapper & trainer
    segmentation — SAM + YOLO segmentation
    depth        — MiDaS depth estimation
    volume       — Point cloud, voxelization, volume estimation
    calorie      — Calorie estimation + XAI (LIME, Counterfactual, Bayesian)
"""

from . import constants
from . import config

__version__ = "0.1.0"
