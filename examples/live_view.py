"""
Live view of the Cinogy CinCam CMOS 1201 (Thorlabs uc480 driver).

Image on a logarithmic beam-profiler colour scale, exposure and gain sliders,
zoom (wheel) and pan (right-drag), the camera readout region (yellow
rectangle, "Apply to camera"), background subtraction, and a save button that
writes the same snapshot folders as snapshot_cincam_uc480.py.

Needs the Thorlabs uc480 driver (ThorCam) and the viewer packages
(pip install -r requirements.txt). Close RayCi / ThorCam first. From the repo
root:

    python examples/live_view.py
"""

from pathlib import Path

from camera_live_view import run_live_view
from cinogy_uc480 import CinCamUC480

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

cam = CinCamUC480({"data_root": DATA_ROOT, "exposure_us": 8000, "gain_db": 0.0})
try:
    run_live_view(cam, DATA_ROOT, experiment="live view")
finally:
    cam.close()
