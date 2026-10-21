"""
Take one snapshot with the Cinogy CinCam CMOS 1201 through the Thorlabs uc480
driver (no RayCi, no pyueye).

Needs the free Thorlabs uc480 driver (install ThorCam). Close RayCi / ThorCam
first: only one program can hold the camera. From the repo root:

    python examples/snapshot_cincam_uc480.py

Saves data/<experiment>/snapshot_<timestamp>/ with data_*.npy (raw frame),
preview_*.png and metadata_*.json.
"""

from pathlib import Path

from cinogy_uc480 import CinCamUC480

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CONFIG = {
    "exposure_us": 8000,
    "gain_db": 0.0,
    "experiment": "smoke_test",
    "comment": "examples/snapshot_cincam_uc480.py",
    "aoi": None,    # readout region, or None for the full sensor:
                    # {"offset_px": (x, y), "width_px": w, "height_px": h}
}

cam = CinCamUC480({"data_root": DATA_ROOT, "exposure_us": CONFIG["exposure_us"], "gain_db": CONFIG["gain_db"]})
try:
    img, out_dir = cam.snapshot(CONFIG["experiment"], comment=CONFIG["comment"], save_preview=True,
                                cropping_params=CONFIG["aoi"])
    peak = int(img.max())
    print(f"{img.shape[1]} x {img.shape[0]} px, peak {peak} / {cam.max_value}"
          + ("  - SATURATED, lower the exposure" if peak >= 0.99 * cam.max_value else ""))
    print(f"saved to {out_dir}")
finally:
    cam.close()
