"""
Cinogy CinCam beam-profiler camera control via the Thorlabs uc480 driver.

Alternative to the `cinogy` package that needs neither RayCi / the CINOGY
driver install nor pyueye: it talks to uc480_64.dll (Thorlabs' build of the
IDS uEye API) through ctypes. Same interface as cinogy.CinCamCMOS1201.

Usage
-----
from cinogy_uc480 import CinCamUC480

cam = CinCamUC480({"data_root": "data", "exposure_us": 6000, "gain_db": 0.0})
img, path = cam.snapshot(experiment_name="alignment", save_preview=True)
cam.close()
"""

from .cincam_uc480 import CinCamUC480

__all__ = ["CinCamUC480"]

__version__ = "0.1.0"
