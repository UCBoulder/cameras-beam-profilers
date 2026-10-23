"""
Analyse a position scan taken with the live viewer's alignment helpers
(data/alignment_data/<name>_<time>/, see camera_live_view/position_scan.py):
Gaussian 1/e² radii against position, parabola fit of w² and the estimated
focal plane.

Edit CONFIG, then run the file (▶ in VS Code, or `python examples/analyze_scan.py`).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from camera_live_view.position_scan import latest_scan
from camera_live_view.scan_analysis import ANALYSES

CONFIG = {
    "folder": None,          # a scan folder, or None for the most recent one under data_root/alignment_data
    "data_root": Path(__file__).resolve().parents[1] / "data",
    "analysis": "Gaussian waist vs position (parabola focus)",
}


def main():
    folder = CONFIG["folder"] or latest_scan(CONFIG["data_root"])
    if folder is None:
        raise SystemExit(f"No scan under {CONFIG['data_root'] / 'alignment_data'}")
    fig = plt.figure(figsize=(8, 5.5))
    fig.canvas.manager.set_window_title(f"{CONFIG['analysis']} — {Path(folder).name}")
    ANALYSES[CONFIG["analysis"]](folder, fig=fig)
    plt.show()


if __name__ == "__main__":
    main()
