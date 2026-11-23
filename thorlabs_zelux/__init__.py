"""
Thorlabs Zelux camera control (CS165MU / CS165MU1) through the Thorlabs
Scientific Camera SDK DLL that ThorImageCAM / ThorCam install, via ctypes.
Same interface as cinogy_uc480.CinCamUC480 and allied_vision.AlviumG1.

Usage
-----
from thorlabs_zelux import ThorlabsZelux

cam = ThorlabsZelux({"data_root": "data", "exposure_us": 6000, "gain_db": 0.0})
img, path = cam.snapshot(experiment_name="alignment", save_preview=True)
cam.close()
"""

from .zelux import ThorlabsZelux

__all__ = ["ThorlabsZelux"]
