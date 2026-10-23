"""
Position-aware snapshots: raw frames taken at known positions (a camera or a
lens moved by hand between shots), kept together in one folder with one
index file. Data taking and analysis are separate: anything that reads the
folder (scan_analysis.py, your own script) gets positions and frames back.

    scan = PositionScan.start(root, "focus scan", unit="mm", start=0.0, step=0.5, camera=cam)
    scan.add(0.0, frame, exposure_us=..., gain_db=..., aoi=...)   # data_+0.000mm.npy
    scan.add(0.5, frame, ...)                                      # data_+0.500mm.npy
    scan.drop_last()                                               # a shot to redo
    folder = scan.finish()

    scan = PositionScan.load(folder)
    for entry in scan.entries:          # sorted by position
        frame = scan.frame(entry)       # uint16 array; entry["position"], entry["exposure_us"], ...

Folder: <root>/alignment_data/<name>_<YYYY_MM_DD-HH_MM_SS>/ with scan.json
(name, unit, start, step, camera metadata, one entry per snapshot) and
data_<position><unit>.npy per snapshot, the position with sign and three
decimals so the files list in order.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

INDEX = "scan.json"


class PositionScan:
    def __init__(self, folder: Path, meta: dict):
        self.folder = Path(folder)
        self.meta = meta

    # ---------- taking data ----------

    @classmethod
    def start(cls, root: str | Path, name: str, unit: str = "mm", start: float = 0.0, step: float = 1.0,
              camera=None) -> "PositionScan":
        """Create <root>/alignment_data/<name>_<stamp>/ and its index."""
        stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        slug = re.sub(r"[^\w-]+", "_", name.strip()) or "scan"
        folder = Path(root) / "alignment_data" / f"{slug}_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        meta = dict(name=name, created=datetime.now().isoformat(timespec="seconds"), unit=unit,
                    start=float(start), step=float(step), finished=None,
                    camera=_camera_info(camera), snapshots=[])
        scan = cls(folder, meta)
        scan._write()
        return scan

    def filename(self, position: float) -> str:
        return f"data_{position:+.3f}{self.meta['unit']}.npy"

    def add(self, position: float, frame: np.ndarray, **info) -> Path:
        """Save `frame` for `position` (a repeated position replaces the earlier shot) and update the index."""
        path = self.folder / self.filename(position)
        np.save(path, np.asarray(frame))
        entry = dict(position=float(position), file=path.name, time=datetime.now().isoformat(timespec="seconds"),
                     shape=list(frame.shape), peak=int(np.max(frame)), **info)
        self.meta["snapshots"] = [e for e in self.meta["snapshots"] if e["file"] != path.name] + [entry]
        self._write()
        return path

    def drop_last(self) -> dict | None:
        """Remove the most recent snapshot (file and entry); returns its entry."""
        if not self.meta["snapshots"]:
            return None
        entry = self.meta["snapshots"].pop()
        (self.folder / entry["file"]).unlink(missing_ok=True)
        self._write()
        return entry

    def finish(self) -> Path:
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self._write()
        return self.folder

    def _write(self):
        (self.folder / INDEX).write_text(json.dumps(self.meta, indent=2, default=_json_safe))

    # ---------- reading data ----------

    @classmethod
    def load(cls, folder: str | Path) -> "PositionScan":
        folder = Path(folder)
        return cls(folder, json.loads((folder / INDEX).read_text()))

    @property
    def entries(self) -> list[dict]:
        return sorted(self.meta["snapshots"], key=lambda e: e["position"])

    @property
    def positions(self) -> np.ndarray:
        return np.array([e["position"] for e in self.entries], dtype=float)

    @property
    def unit(self) -> str:
        return self.meta.get("unit", "")

    def frame(self, entry: dict) -> np.ndarray:
        return np.load(self.folder / entry["file"])

    def __len__(self):
        return len(self.meta["snapshots"])


def latest_scan(root: str | Path) -> Path | None:
    """The most recently created scan folder under <root>/alignment_data, if any."""
    folders = [p for p in (Path(root) / "alignment_data").glob("*") if (p / INDEX).exists()]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def _camera_info(camera) -> dict:
    """JSON-safe description of the camera (model, serial, pixel size, dead-pixel map, ...)."""
    if camera is None:
        return {}
    info = dict(getattr(camera, "_camera_metadata", {}) or {})
    info.setdefault("class", type(camera).__name__)
    for key in ("pixel_size_um", "sensor_width", "sensor_height", "serial", "max_value"):
        value = getattr(camera, key, None)
        if value is not None:
            info.setdefault(key, value)
    dead = getattr(camera, "dead_pixel_file", None)
    if dead is not None and Path(dead).exists():
        info["dead_pixel_file"] = str(dead)
    return info


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
