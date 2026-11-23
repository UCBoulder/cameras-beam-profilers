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
data_<position><unit>.npy per snapshot, the position with sign and at least
three decimals (up to six, format_position()), so that sub-µm positions in mm
get their own files.

Several positions per snapshot (e.g. a target z and a camera z):

    scan = PositionScan.start(root, "PSF vs z", subfolder="experiment_data", axes=("target_z", "camera_z"))
    scan.add({"target_z": 0.0, "camera_z": 1.5}, frame, ...)       # data_target_z+0.000mm_camera_z+1.500mm.npy
    scan.add_background(dark_frame, exposure_us=..., gain_db=..., aoi=...)   # background_<exposure>us_<gain>dB.npy
    entry["positions"]["camera_z"]; scan.backgrounds; scan.background_for(entry)

A snapshot or background taken again at the same position (or with the same
exposure and gain) replaces the earlier one.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

INDEX = "scan.json"
POSITION_DECIMALS = 6               # positions are typed and named to 1e-6 of their unit (1 nm in mm)


def format_position(value: float) -> str:
    """'+12.700', '+0.000450': sign, at least three decimals, up to POSITION_DECIMALS (no trailing zeros beyond)."""
    whole, _, decimals = f"{value:+.{POSITION_DECIMALS}f}".rstrip("0").partition(".")
    return f"{whole}.{decimals.ljust(3, '0')}"


class PositionScan:
    def __init__(self, folder: Path, meta: dict):
        self.folder = Path(folder)
        self.meta = meta

    # ---------- taking data ----------

    @classmethod
    def start(cls, root: str | Path, name: str, unit: str = "mm", start: float = 0.0, step: float = 1.0,
              camera=None, subfolder: str = "alignment_data", axes: tuple[str, ...] | None = None,
              **info) -> "PositionScan":
        """
        Create <root>/<subfolder>/<name>_<stamp>/ and its index (`info` is stored in it
        as is). With `axes`, each snapshot has one position per axis (a dict).
        """
        stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        slug = re.sub(r"[^\w-]+", "_", name.strip()) or "scan"
        folder = Path(root) / subfolder / f"{slug}_{stamp}"
        for n in range(2, 100):                      # two scans started within the same second
            if not folder.exists():
                break
            folder = folder.with_name(f"{slug}_{stamp}_{n}")
        folder.mkdir(parents=True, exist_ok=False)
        meta = dict(name=name, created=datetime.now().isoformat(timespec="seconds"), unit=unit,
                    start=float(start), step=float(step), finished=None, **info,
                    camera=_camera_info(camera), snapshots=[])
        if axes:
            meta["axes"] = list(axes)
            meta["backgrounds"] = []
        scan = cls(folder, meta)
        scan._write()
        return scan

    def filename(self, position: float | dict) -> str:
        unit = re.sub(r"[^\w]+", "", self.meta["unit"])     # file-name safe ("inch", "mm", "µm", "steps")
        if isinstance(position, dict):
            return "data_" + "_".join(f"{axis}{format_position(position[axis])}{unit}" for axis in self.axes) + ".npy"
        return f"data_{format_position(position)}{unit}.npy"

    @property
    def axes(self) -> list[str]:
        """Names of the position axes of a multi-axis scan ([] for a one-axis scan)."""
        return list(self.meta.get("axes", []))

    def add(self, position: float | dict, frame: np.ndarray, **info) -> Path:
        """Save `frame` for `position` (a repeated position replaces the earlier shot) and update the index."""
        path = self.folder / self.filename(position)
        np.save(path, np.asarray(frame))
        if isinstance(position, dict):
            missing = set(self.axes) - set(position)
            if missing:
                raise ValueError(f"position lacks the axes {sorted(missing)}")
            where = dict(positions={axis: float(position[axis]) for axis in self.axes})
        else:
            where = dict(position=float(position))
        entry = dict(**where, file=path.name, time=datetime.now().isoformat(timespec="seconds"),
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

    def add_background(self, frame: np.ndarray, exposure_us: float | None = None, gain_db: float | None = None,
                       **info) -> Path:
        """Save a background frame (PSF light blocked); one per exposure and gain, a repeat replaces it."""
        name = "background" + (f"_{exposure_us:.0f}us" if exposure_us is not None else "") \
            + (f"_{gain_db:.1f}dB" if gain_db is not None else "") + ".npy"
        path = self.folder / name
        np.save(path, np.asarray(frame))
        entry = dict(file=name, time=datetime.now().isoformat(timespec="seconds"), shape=list(frame.shape),
                     peak=int(np.max(frame)), exposure_us=exposure_us, gain_db=gain_db, **info)
        self.meta["backgrounds"] = [b for b in self.meta.get("backgrounds", []) if b["file"] != name] + [entry]
        self._write()
        return path

    @property
    def backgrounds(self) -> list[dict]:
        return list(self.meta.get("backgrounds", []))

    def background_for(self, entry: dict) -> dict | None:
        """The background taken with the same exposure and gain as `entry` (and covering its readout), or None."""
        for b in self.backgrounds:
            if _close(b.get("exposure_us"), entry.get("exposure_us")) and _close(b.get("gain_db"), entry.get("gain_db")):
                return b
        return None

    def missing_backgrounds(self) -> list[tuple[float | None, float | None]]:
        """(exposure_us, gain_db) settings used by snapshots that have no matching background yet."""
        missing = []
        for e in self.meta["snapshots"]:
            key = (e.get("exposure_us"), e.get("gain_db"))
            if self.background_for(e) is None and key not in missing:
                missing.append(key)
        return missing

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
        if self.axes:
            return sorted(self.meta["snapshots"], key=lambda e: tuple(e["positions"][a] for a in self.axes))
        return sorted(self.meta["snapshots"], key=lambda e: e["position"])

    @property
    def positions(self) -> np.ndarray:
        """Positions in entry order: (n,) for a one-axis scan, (n, n_axes) for a multi-axis scan."""
        if self.axes:
            return np.array([[e["positions"][a] for a in self.axes] for e in self.entries], dtype=float)
        return np.array([e["position"] for e in self.entries], dtype=float)

    @property
    def unit(self) -> str:
        return self.meta.get("unit", "")

    def frame(self, entry: dict) -> np.ndarray:
        """The saved frame of a snapshot or background entry."""
        return np.load(self.folder / entry["file"])

    def __len__(self):
        return len(self.meta["snapshots"])


def latest_scan(root: str | Path, subfolder: str = "alignment_data") -> Path | None:
    """The most recently created scan folder under <root>/<subfolder>, if any."""
    folders = [p for p in (Path(root) / subfolder).glob("*") if (p / INDEX).exists()]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.isclose(float(a), float(b), rtol=1e-3, atol=1e-6))


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
