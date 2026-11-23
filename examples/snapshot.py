"""
Take one snapshot. Pick the camera in CONFIG or on the command line, from the
repo root:

    python examples/snapshot.py                # CONFIG["camera"]
    python examples/snapshot.py cincam         # Cinogy CinCam through the Thorlabs uc480 driver
    python examples/snapshot.py cincam_ueye    # Cinogy CinCam through the CINOGY / RayCi driver
    python examples/snapshot.py alvium         # Allied Vision through Vimba X
    python examples/snapshot.py zelux          # Thorlabs Zelux through the ThorImageCAM / ThorCam SDK

Close the vendor's own viewer first (ThorCam / ThorImageCAM / RayCi / Vimba X Viewer): only
one program can hold a camera. Allied Vision settings are in cameras.py.

Saves <DATA_ROOT>/<experiment>/snapshot_<timestamp>/ (DATA_ROOT in cameras.py)
with data_*.npy (raw frame), preview_*.png and metadata_*.json.
"""

import sys

from cameras import DATA_ROOT, open_camera

CONFIG = {
    "camera": "cincam",     # "cincam", "cincam_ueye", "alvium" or "zelux"
    "exposure_us": 8000,
    "gain_db": 0.0,
    "experiment": "smoke_test",
    "comment": "examples/snapshot.py",
    "aoi": None,    # readout region, or None for the full sensor:
                    # {"offset_px": (x, y), "width_px": w, "height_px": h}
}

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else CONFIG["camera"]
    cam = open_camera(kind, DATA_ROOT, CONFIG["exposure_us"], CONFIG["gain_db"])
    try:
        img, out_dir = cam.snapshot(CONFIG["experiment"], comment=CONFIG["comment"], save_preview=True,
                                    cropping_params=CONFIG["aoi"])
        peak = int(img.max())
        print(f"{img.shape[1]} x {img.shape[0]} px, peak {peak} / {cam.max_value}"
              + ("  - SATURATED, lower the exposure" if peak >= 0.99 * cam.max_value else ""))
        print(f"saved to {out_dir}")
    finally:
        cam.close()
