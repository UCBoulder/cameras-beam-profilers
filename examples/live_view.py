"""
Live view of a beam-profiler camera: image on a logarithmic beam-profiler
colour scale, exposure and gain sliders, zoom (wheel) and pan (right-drag),
the camera readout region (yellow rectangle, "Apply to camera"), background
subtraction, live 2D Gaussian fit, dead-pixel map, alignment helpers, and a
save button that writes the same snapshot folders as the snapshot examples.

Pick the camera in CONFIG or on the command line:

    python examples/live_view.py            # CONFIG["camera"]
    python examples/live_view.py cincam     # Cinogy CinCam through the Thorlabs uc480 driver
    python examples/live_view.py alvium     # Allied Vision (Alvium, Mako over GigE) through Vimba X

Close the vendor's own viewer first (ThorCam / RayCi / Vimba X Viewer): only
one program can hold a camera.
"""

import sys
from pathlib import Path

from camera_live_view import run_live_view

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CONFIG = {
    "camera": "cincam",          # "cincam" or "alvium"
    "exposure_us": 8000,
    "gain_db": 0.0,
    "experiment": "live view",   # default experiment name in the save dialog
    "alvium": {                  # extra settings for Allied Vision cameras
        "camera_id": None,       # None: the first real camera found (Vimba X simulators are skipped)
        "pixel_size_um": None,   # e.g. 3.45 for a Mako G-131B; Vimba does not report it
        "use_camera_roi": False, # start with the full sensor
    },
}


def open_camera(kind: str):
    config = {"data_root": DATA_ROOT, "exposure_us": CONFIG["exposure_us"], "gain_db": CONFIG["gain_db"]}
    if kind == "cincam":
        from cinogy_uc480 import CinCamUC480
        return CinCamUC480(config)
    if kind == "alvium":
        from allied_vision import AlviumG1
        return AlviumG1({**config, **CONFIG["alvium"]})
    raise SystemExit(f"Unknown camera '{kind}': use 'cincam' or 'alvium'")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else CONFIG["camera"]
    cam = open_camera(kind)
    try:
        run_live_view(cam, DATA_ROOT, experiment=CONFIG["experiment"])
    finally:
        cam.close()
