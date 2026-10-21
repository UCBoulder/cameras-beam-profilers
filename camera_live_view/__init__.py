"""
Live viewer for cameras with a live mode (cinogy_uc480.CinCamUC480): image on
a logarithmic beam-profiler colour scale, exposure / gain controls, zoom,
camera readout region, background subtraction and a snapshot button.
Needs pyqtgraph and PySide6 (pip install -e ".[live]").

    from camera_live_view import run_live_view
    run_live_view(cam, data_root="data", experiment="alignment")
"""

from .live_view import LiveView, SnapshotDialog, colour_map, run_live_view

__all__ = ["LiveView", "SnapshotDialog", "colour_map", "run_live_view"]
