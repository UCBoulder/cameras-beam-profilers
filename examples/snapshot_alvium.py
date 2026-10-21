"""
Take one snapshot with an Allied Vision camera (Alvium G1, Mako G-131B, ...)
through vmbpy.

Needs the Vimba X SDK (see the README). Close Vimba X Viewer first: only one
program can hold the camera. From the repo root:

    python examples/snapshot_alvium.py

Saves data/<experiment>/snapshot_<timestamp>/ with data_*.npy (raw frame),
preview_*.png and metadata_*.json.
"""

from pathlib import Path

from allied_vision import AlviumG1

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CONFIG = {
    "exposure_us": 6000,
    "gain_db": 0.0,
    "experiment": "smoke_test",
    "comment": "examples/snapshot_alvium.py",
    "aoi": None,    # readout region, or None for the full sensor:
                    # {"offset_px": (x, y), "width_px": w, "height_px": h}
}

cam = AlviumG1({"data_root": DATA_ROOT, "exposure_us": CONFIG["exposure_us"], "gain_db": CONFIG["gain_db"]})
try:
    img, out_dir = cam.snapshot(CONFIG["experiment"], comment=CONFIG["comment"], save_preview=True,
                                cropping_params=CONFIG["aoi"])
    peak = int(img.max())
    print(f"{img.shape[1]} x {img.shape[0]} px, peak {peak} / {cam.max_value}"
          + ("  - SATURATED, lower the exposure" if peak >= 0.99 * cam.max_value else ""))
    print(f"saved to {out_dir}")
finally:
    cam.close()
