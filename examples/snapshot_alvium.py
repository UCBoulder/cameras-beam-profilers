"""
Minimal hardware smoke test: connect, take one snapshot, disconnect.

Requires a camera attached and the Vimba X SDK installed. Run from the repo
root after `pip install -e .`:

    python examples/snapshot.py
"""

from pathlib import Path

from allied_vision import AlviumG1

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

config = {
    "data_root": DATA_ROOT,
    "exposure_us": None,   # None -> falls back to the _PARAMETERS default (6000)
    "gain_db": None,       # None -> falls back to the _PARAMETERS default (0.0)
}

cam = AlviumG1(config)

img, out_dir = cam.snapshot(
    experiment_name="smoke_test",
    comment="examples/snapshot.py",
    save_preview=True,
    filename="test",
)

print(f"Captured {img.shape} {img.dtype}, saved to {out_dir}")

cam.close()
