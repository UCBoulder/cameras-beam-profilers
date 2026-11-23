"""
Live view of a beam-profiler camera: image on a logarithmic beam-profiler
colour scale, exposure and gain sliders, zoom (wheel) and pan (right-drag),
the camera readout region (yellow rectangle, "Apply to camera"), background
subtraction, live 2D Gaussian fit, dead-pixel map, alignment helpers, and a
save button that writes the same snapshot folders as snapshot.py.

Pick the camera in CONFIG or on the command line:

    python examples/live_view.py            # CONFIG["camera"]
    python examples/live_view.py cincam     # Cinogy CinCam through the Thorlabs uc480 driver
    python examples/live_view.py alvium     # Allied Vision (Alvium, Mako over GigE) through Vimba X
    python examples/live_view.py zelux      # Thorlabs Zelux through the ThorImageCAM / ThorCam SDK

Close the vendor's own viewer first (ThorCam / ThorImageCAM / RayCi / Vimba X Viewer): only
one program can hold a camera. Allied Vision settings are in cameras.py.
"""

import sys

from camera_live_view import run_live_view
from cameras import DATA_ROOT, open_camera

CONFIG = {
    "camera": "zelux",          # "cincam", "alvium" or "zelux" (cincam_ueye has no live mode)
    "exposure_us": 3000,
    "gain_db": 0.0,
    "experiment": "live view",   # default experiment name in the save dialog
}


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else CONFIG["camera"]
    cam = open_camera(kind, DATA_ROOT, CONFIG["exposure_us"], CONFIG["gain_db"])
    try:
        run_live_view(cam, DATA_ROOT, experiment=CONFIG["experiment"])
    finally:
        cam.close()
