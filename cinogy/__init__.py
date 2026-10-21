"""
Cinogy CinCam beam-profiler camera control for the yb2 lab.

Snapshot-oriented wrapper around the IDS uEye API (pyueye), with the same
interface as allied_vision.AlviumG1.

Usage
-----
from cinogy import CinCamCMOS1201

cam = CinCamCMOS1201({"data_root": "data", "exposure_us": 6000, "gain_db": 0.0})
img, path = cam.snapshot(experiment_name="alignment", save_preview=True)
cam.close()
"""

from .cincam_cmos import CinCamCMOS1201

__all__ = ["CinCamCMOS1201"]

__version__ = "0.1.0"
