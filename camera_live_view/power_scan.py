"""
Power-meter scan: where on the camera is the transmitted power highest?

The beam is moved slowly across something that transmits it to a power meter
(an aperture, a fibre, a pinhole) while the camera watches it. Every live 2D
Gaussian fit gives the spot centre (x, y) on the sensor; it is logged with
the power read during that frame. A scan runs along x or along y; when it is
stopped, a Gaussian on an offset is fitted to the power against the centre,

    P(s) = A exp(−2 (s − s0)² / w²) + B,      s = x or y (sensor px),

and s0 is the camera position of maximum power. With s0 from an x scan and
one from a y scan, (x0, y0) can be marked as a reference point in the viewer.

Nothing here depends on the camera or the power meter: the samples are fit
centres (any camera of the viewer) and readings from any object with the
power-meter interface: power_w() (one reading in W, may block), close(),
description (str), and optionally info() (dict for the saved metadata),
wavelength_nm / set_wavelength_nm(). thorlabs_power_meter.ThorlabsPowerMeter
is one.

    reader = PowerReader(meter)               # reads in a background thread
    power, n = reader.mean_since(t_start)     # mean of the readings since t_start
    fit = fit_gaussian_offset(x, power)       # dict(s0, s0_err, w, ...), None if it fails
    folder = save_power_scan(root, "power scan", "x", samples, fit, camera=cam, meter_info=meter.info())
    meta, samples = load_power_scan(folder)

Folder (as the other alignment helpers): <root>/<name>_<axis>_<YYYY_MM_DD-HH_MM_SS>/
with scan.json (name, axis, camera and power-meter metadata, the fit),
power_scan.csv (one row per sample) and power_vs_<axis>.png.
"""

import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from .position_scan import PositionScan

DATA_FILE = "power_scan.csv"
COLUMNS = ("t_s", "x_px", "y_px", "power_w", "n_readings", "fit_peak_counts", "wx_px", "wy_px")
MIN_POINTS = 5
MIN_R2 = 0.5            # the Gaussian must explain at least this fraction of the variance of the power


def gaussian_offset(s, amplitude, s0, w, offset):
    """Gaussian with 1/e² radius w on a constant offset."""
    return amplitude * np.exp(-2.0 * (s - s0) ** 2 / w ** 2) + offset


class PowerReader:
    """
    Reads meter.power_w() in a background thread, as fast as the meter answers
    (its averaging time sets the rate), and keeps the last KEEP_S seconds of
    (time.perf_counter(), W). The meter is only called from that thread or
    through call(), one call at a time.
    """

    KEEP_S = 30.0

    def __init__(self, meter):
        self.meter = meter
        self.error = None                   # text of the last failed reading, None while readings arrive
        self._meter_lock, self._data_lock = threading.Lock(), threading.Lock()
        self._readings = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="power meter", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                with self._meter_lock:
                    power = float(self.meter.power_w())
            except Exception as err:        # unplugged, driver error: keep trying, report it
                self.error = f"{type(err).__name__}: {err}"
                self._stop.wait(0.5)
                continue
            self.error = None
            now = time.perf_counter()
            with self._data_lock:
                self._readings.append((now, power))
                while self._readings[0][0] < now - self.KEEP_S:
                    self._readings.popleft()
            self._stop.wait(0.002)          # lets call() in between readings

    def call(self, function, *args):
        """Run function(*args) (e.g. meter.set_wavelength_nm) between two readings."""
        with self._meter_lock:
            return function(*args)

    def latest(self) -> tuple[float, float] | None:
        """(time, W) of the newest reading, None before the first."""
        with self._data_lock:
            return self._readings[-1] if self._readings else None

    def mean_since(self, t0: float) -> tuple[float | None, int]:
        """Mean power of the readings taken since t0 (perf_counter) and how many there were; the newest
        reading (count 1) when the meter was slower than that, (None, 0) before the first reading."""
        with self._data_lock:
            values = [p for t, p in self._readings if t >= t0]
            if not values and self._readings:
                values = [self._readings[-1][1]]
        return (float(np.mean(values)), len(values)) if values else (None, 0)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)


def fit_gaussian_offset(s, power, start: dict | None = None) -> dict | None:
    """
    gaussian_offset() fitted to power against position s (least squares, A ≥ 0,
    w ≥ 0.1 px, s0 within a scan length of the data): dict(s0, s0_err, w,
    w_err, amplitude, offset, peak_power, n, inside, rms). `start` (a previous
    result) gives the start values; otherwise the offset starts at the lowest
    power, s0 at the highest, w from the second moment of the power above the
    offset. inside: s0 lies within the scanned range (else the maximum was not
    reached). r2: the fraction of the variance of the power the Gaussian
    explains. None with fewer than MIN_POINTS points, no spread in s, if the
    fit fails, or if r2 < MIN_R2 (no peak to speak of: flat or pure noise, e.g.
    the beam never reached the aperture).
    """
    s, p = np.asarray(s, float), np.asarray(power, float)
    ok = np.isfinite(s) & np.isfinite(p)
    s, p = s[ok], p[ok]
    if len(s) < MIN_POINTS or np.ptp(s) == 0:
        return None
    span = float(np.ptp(s))
    if start:
        p0 = [start["amplitude"], start["s0"], start["w"], start["offset"]]
    else:
        offset = float(p.min())
        above = np.clip(p - offset, 0, None)
        centre = float(s[np.argmax(p)])
        w = 2 * np.sqrt(np.sum(above * (s - centre) ** 2) / above.sum()) if above.sum() > 0 else span / 4
        p0 = [max(float(p.max()) - offset, 1e-12), centre, float(np.clip(w, 0.5, span)), offset]
    lower = [0.0, s.min() - span, 0.1, -np.inf]
    upper = [np.inf, s.max() + span, 10 * span, np.inf]
    p0 = list(np.clip(p0, lower, upper))
    scale = max(float(np.ptp(p)), float(np.abs(p).max()), 1e-15)     # powers in W are tiny: fit in units of it
    try:
        popt, pcov = curve_fit(gaussian_offset, s, p / scale, p0=[p0[0] / scale, p0[1], p0[2], p0[3] / scale],
                               bounds=([lower[0], lower[1], lower[2], -np.inf], [np.inf, upper[1], upper[2], np.inf]),
                               maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    err = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan)
    amplitude, s0, w, offset = float(popt[0] * scale), float(popt[1]), float(popt[2]), float(popt[3] * scale)
    rms = float(np.sqrt(np.mean((gaussian_offset(s, amplitude, s0, w, offset) - p) ** 2)))
    r2 = 1.0 - rms ** 2 / p.var() if p.var() > 0 else 0.0
    if r2 < MIN_R2:
        return None
    return dict(s0=s0, s0_err=float(err[1]), w=w, w_err=float(err[2]), amplitude=amplitude, offset=offset,
                peak_power=amplitude + offset, n=int(len(s)), inside=bool(s.min() <= s0 <= s.max()), rms=rms,
                r2=float(r2))


def format_power(watts: float | None) -> str:
    """'12.34 µW' etc."""
    if watts is None or not np.isfinite(watts):
        return "—"
    for factor, unit in ((1.0, "W"), (1e-3, "mW"), (1e-6, "µW"), (1e-9, "nW")):
        if abs(watts) >= factor:
            return f"{watts / factor:.4g} {unit}"
    return f"{watts / 1e-12:.4g} pW"


def power_scan_figure(samples: list[dict], axis: str, fit: dict | None, title: str = "") -> Figure:
    """Power against the fitted spot centre with the Gaussian and its maximum."""
    s = np.array([r[f"{axis}_px"] for r in samples], float)
    p = np.array([r["power_w"] for r in samples], float)
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot()
    scale, unit = next(((f, u) for f, u in ((1.0, "W"), (1e-3, "mW"), (1e-6, "µW"), (1e-9, "nW"))
                        if np.nanmax(np.abs(p), initial=0) >= f), (1e-12, "pW"))
    ax.plot(s, p / scale, "o", ms=4, alpha=0.7, label=f"{len(s)} samples")
    if fit:
        grid = np.linspace(min(s.min(), fit["s0"] - 2 * fit["w"]), max(s.max(), fit["s0"] + 2 * fit["w"]), 400)
        ax.plot(grid, gaussian_offset(grid, fit["amplitude"], fit["s0"], fit["w"], fit["offset"]) / scale, "-", lw=2,
                label=f"Gaussian + offset: w = {fit['w']:.1f} px")
        ax.axvline(fit["s0"], color="C3", ls="--",
                   label=f"maximum at {axis} = {fit['s0']:.2f} ± {fit['s0_err']:.2f} px"
                         + ("" if fit["inside"] else " (outside the scan)"))
    ax.set_xlabel(f"spot centre {axis} on the sensor (px, top-left pixel 0)")
    ax.set_ylabel(f"power ({unit})")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save_power_scan(root: str | Path, name: str, axis: str, samples: list[dict], fit: dict | None,
                    camera=None, meter_info: dict | None = None, **info) -> Path:
    """Save a finished scan (see the module docstring); returns its folder."""
    scan = PositionScan.start(root, f"{name} {axis}", unit="px", camera=camera, subfolder="", kind="power-meter scan",
                              axis=axis, power_meter=meter_info or {}, data_file=DATA_FILE, n_samples=len(samples),
                              gaussian_fit=fit, **info)
    for unused in ("start", "step"):       # a position scan's typed start and step: none here
        scan.meta.pop(unused, None)
    with open(scan.folder / DATA_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    fig = power_scan_figure(samples, axis, fit, f"{scan.folder.name}\n{(meter_info or {}).get('device', '')}")
    fig.savefig(scan.folder / f"power_vs_{axis}.png", dpi=130)
    return scan.finish()


def load_power_scan(folder: str | Path) -> tuple[dict, np.ndarray]:
    """(scan.json contents, samples as a structured array with the COLUMNS as fields)."""
    folder = Path(folder)
    meta = PositionScan.load(folder).meta
    samples = np.genfromtxt(folder / meta.get("data_file", DATA_FILE), delimiter=",", names=True, ndmin=1)
    return meta, samples


__all__ = ["PowerReader", "fit_gaussian_offset", "format_power", "gaussian_offset", "load_power_scan",
           "power_scan_figure", "save_power_scan"]
