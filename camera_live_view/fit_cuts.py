"""
Live horizontal and vertical cuts through the spot at the centre of the 2D
Gaussian fit, in a window of their own to put next to the image (live viewer:
Gaussian fit → "x / y cuts through the fit centre…").

The x cut is the pixel row nearest the fitted centre, the y cut the pixel
column, both over ± the chosen number of pixels around the centre, read
straight from the frame the fit saw (background subtracted and dead pixels
repaired as set in the viewer): no interpolation. The positions are measured
from the fitted (sub-pixel) centre. The fitted 2D Gaussian along the same row
and column is drawn dashed.

The y axis stays where it is (full scale, or 0 … 1.25 normalised; zoom with
the mouse and it stays there) so a changing peak can be read off. A marker at the
maximum of each cut prints its raw value. "Normalise to the peak" divides each
cut and its Gaussian by the maximum of the cut: the first ring then reads
directly as a fraction of the peak (1.75 % for a perfect Airy pattern), best
seen with the log y axis.

Ring / peak: the central lobe |s| < r0 is ignored (grey band), r0 = 0.61 λ/NA
the first zero of the Airy pattern, J1(k r NA) = 0, in camera pixels
(0.61 λ/NA × magnification / pixel size; the viewer fills it in from the
launcher's optics, or type it). The highest point of each cut outside the band
(marked ◆) is printed as a fraction of the peak, both with the fitted constant
offset removed so a dark level does not read as a ring.

Central-lobe fit: a 1D Gaussian with offset, A exp(−2 (s − s0)² / w²) + B, is
fitted to the points of each cut with |s| ≤ r0 only (drawn dotted inside the
grey band). The table on the right lists, per cut, its maximum, the fitted 1/e²
radius w, and with the imaging system set ("Imaging system…": wavelength,
magnification, nominal NA, camera pixel; the viewer keeps it per camera) w on
the object and the NA = 0.421 λ / w, the pixel's widening removed in
quadrature as in Objective-testing's PSFAnalyzer.na_from_waist_analytic().
The nominal NA sets r0 = 0.61 λ/NA, so the fit window follows the optics.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.optimize import curve_fit

from .gaussian_fit import gaussian_2d

COLOURS = {"x": "#4da6ff", "y": "#ff5c5c"}
AIRY_FIRST_RING = 0.0175          # first-ring maximum of a perfect Airy pattern / its peak
AIRY_FIRST_ZERO = 0.6098          # first zero r0 = 0.61 λ / NA (J1(x) = 0 at x = 3.8317)
AIRY_W_PER_LAMBDA_NA = 0.4208     # Gaussian least-squares fitted to the Airy central lobe: w = 0.421 λ / NA

# Imaging system (Imaging system… dialog): key -> (label, suffix, maximum, decimals, tooltip); 0 = not set
OPTICS_FIELDS = {
    "wavelength_nm": ("Wavelength", " nm", 3000.0, 1, "Illumination wavelength"),
    "magnification": ("Magnification", "", 10000.0, 3, "Total magnification from the object (pinhole) plane "
                                                       "to the camera"),
    "na": ("Nominal NA", "", 1.6, 3, "NA the objective should have: sets the first zero r0 = 0.61 λ/NA, the "
                                     "window of the central-lobe fit"),
    "pixel_size_um": ("Camera pixel", " µm", 100.0, 3, "Pixel pitch on the sensor (CinCam 5.2 µm, Mako G-131B "
                                                       "5.3 µm, Mako G-319B 3.45 µm)"),
}


def first_zero_px(optics: dict | None) -> float | None:
    """r0 = 0.61 λ/NA on the camera in pixels, from optics (wavelength_nm, na, magnification,
    pixel_size_um); None if unknown."""
    try:
        r0_um = AIRY_FIRST_ZERO * optics["wavelength_nm"] * 1e-3 / optics["na"]
        return r0_um * optics["magnification"] / optics["pixel_size_um"]
    except (TypeError, KeyError, ZeroDivisionError):
        return None


def na_from_lobe(w_px: float, w_err_px: float, optics: dict) -> dict | None:
    """
    NA from the 1/e² radius of the central-lobe fit, as PSFAnalyzer.na_from_waist_analytic()
    in Objective-testing: the Gaussian is fitted to the pixel values (not pixel-averaged), so
    the pixel's widening is removed in quadrature, σ² = (w/2)² − p²/12 with p = 1 px, and
    NA = 0.421 λ / w. dict(w_nm, w_err_nm, na, na_err), errors from the fit of w only; None
    without the optics.
    """
    try:
        px_nm = 1000.0 * optics["pixel_size_um"] / optics["magnification"]   # one camera pixel on the object
        wavelength_nm = float(optics["wavelength_nm"])
    except (TypeError, KeyError, ZeroDivisionError):
        return None
    sigma2 = (w_px / 2.0) ** 2 - 1.0 / 12.0
    if sigma2 <= 0 or wavelength_nm <= 0:
        return None
    w_spot = 2.0 * np.sqrt(sigma2)                       # px, pixel widening removed
    na = AIRY_W_PER_LAMBDA_NA * wavelength_nm / (w_spot * px_nm)
    w_err_spot = w_err_px * w_px / w_spot                # d(w_spot)/dw = w / w_spot
    return dict(w_nm=w_spot * px_nm, w_err_nm=w_err_spot * px_nm, na=na, na_err=na * w_err_spot / w_spot)


class OpticsDialog(QtWidgets.QDialog):
    """The imaging system for the central-lobe fit: its window r0 and the NA from its width."""

    def __init__(self, optics: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Imaging system")
        form = QtWidgets.QFormLayout(self)
        self.spins = {}
        for key, (label, suffix, maximum, decimals, tip) in OPTICS_FIELDS.items():
            spin = QtWidgets.QDoubleSpinBox(decimals=decimals, suffix=suffix)
            spin.setRange(0.0, maximum)
            spin.setSpecialValueText("not set")
            spin.setValue(float(optics.get(key) or 0.0))
            spin.setToolTip(tip)
            form.addRow(label, spin)
            self.spins[key] = spin
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        return {key: (spin.value() or None) for key, spin in self.spins.items()}


def gaussian_1d(s, amplitude, s0, w, offset):
    """1D Gaussian with 1/e² radius w on a constant offset."""
    return amplitude * np.exp(-2.0 * (s - s0) ** 2 / w ** 2) + offset


def lobe_fit(s: np.ndarray, values: np.ndarray, r0: float, w_start: float, offset: float | None = None) -> dict | None:
    """
    gaussian_1d() fitted to the points with |s| ≤ r0 (the central lobe): dict(amplitude,
    s0, w, offset, peak, w_err); None with fewer than 5 points or if the fit fails.
    offset=None fits the offset too; a number holds it there (e.g. the background
    level measured in the wings). Inside r0 alone a free offset is poorly pinned: it
    absorbs the difference between the lobe (zero at r0) and a Gaussian, and can
    come out tens of counts negative with w biased up.
    """
    sel = (np.abs(s) <= r0) & np.isfinite(values)
    if sel.sum() < 5:
        return None
    x, y = s[sel], values[sel]
    start = [max(y.max() - y.min(), 1e-6), float(x[np.argmax(y)]), float(np.clip(w_start, 0.5, 2 * r0))]
    lower, upper = [0.0, -r0, 0.3], [np.inf, r0, 4 * r0]
    if offset is None:
        model = gaussian_1d
        start, lower, upper = start + [float(y.min())], lower + [-np.inf], upper + [np.inf]
    else:
        def model(t, amplitude, s0, w):
            return gaussian_1d(t, amplitude, s0, w, offset)
    try:
        popt, pcov = curve_fit(model, x, y, p0=start, maxfev=2000, bounds=(lower, upper))
    except (RuntimeError, ValueError):
        return None
    amplitude, s0, w = (float(v) for v in popt[:3])
    b = float(popt[3]) if offset is None else float(offset)
    w_err = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else float("nan")
    return dict(amplitude=amplitude, s0=s0, w=w, offset=b, peak=amplitude + b, w_err=w_err)


def centre_cuts(frame: np.ndarray, fit: dict, half_length: int) -> dict:
    """
    The row (x cut) and column (y cut) of `frame` through the pixel nearest the
    fitted centre (array coordinates, pixel i centred at i), over ±half_length
    pixels: {"x": dict(s, values, s_fine, gaussian, line, w), "y": ...}, s measured
    from the fitted centre, w the 1/e² radius of the fitted Gaussian along the cut.
    Empty when the centre lies off the frame.
    """
    img = np.asarray(frame, dtype=float)
    h, w = img.shape
    ix, iy = int(round(fit["x0"])), int(round(fit["y0"]))
    p = (fit["amplitude"], fit["x0"], fit["y0"], fit["wx"], fit["wy"], fit["theta"], fit["offset"])
    c2, s2 = np.cos(fit["theta"]) ** 2, np.sin(fit["theta"]) ** 2
    cuts = {}
    if not (0 <= ix < w and 0 <= iy < h):
        return cuts
    for name, centre, centre_px, size, w_cut in (
            ("x", fit["x0"], ix, w, 1 / np.sqrt(c2 / fit["wx"] ** 2 + s2 / fit["wy"] ** 2)),
            ("y", fit["y0"], iy, h, 1 / np.sqrt(s2 / fit["wx"] ** 2 + c2 / fit["wy"] ** 2))):
        idx = np.arange(max(centre_px - half_length, 0), min(centre_px + half_length + 1, size))
        fine = np.linspace(idx[0], idx[-1], 10 * idx.size)
        if name == "x":
            values, gaussian, line = img[iy, idx], gaussian_2d((fine, np.full_like(fine, iy)), *p), iy
        else:
            values, gaussian, line = img[idx, ix], gaussian_2d((np.full_like(fine, ix), fine), *p), ix
        cuts[name] = dict(s=idx - centre, values=values, s_fine=fine - centre, gaussian=gaussian, line=line, w=w_cut,
                          offset=fit["offset"])
    return cuts


class FitCutsWindow(QtWidgets.QWidget):
    """The cut window; show_cuts() with every new fit. opticsChanged: the imaging system was edited."""

    opticsChanged = QtCore.Signal(dict)

    def __init__(self, max_value: float, parent=None, optics: dict | None = None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)    # own window, closes with the viewer
        self.setWindowTitle("x / y cuts through the Gaussian fit centre")
        self.resize(1050, 560)
        self.max_value = float(max_value)
        self.optics = dict(optics or {})                          # OPTICS_FIELDS keys
        r0_px = first_zero_px(self.optics)
        self._last = None                                         # (frame, fit) of the last update

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "distance from the fitted centre (px)")
        self.plot.addLegend(offset=(-10, 10), brush=(0, 0, 0, 170))
        self.items, self.checks = {}, {}
        for name in COLOURS:
            colour = COLOURS[name]
            self.items[name] = dict(
                data=self.plot.plot([], [], pen=pg.mkPen(colour, width=2), symbol="o", symbolSize=5,
                                    symbolBrush=colour, symbolPen=None, connect="finite", name=f"{name} cut"),
                gaussian=self.plot.plot([], [], pen=pg.mkPen(colour, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                                        connect="finite"),
                marker=self.plot.plot([], [], pen=None, symbol="t", symbolSize=14, symbolBrush=colour,
                                      symbolPen=pg.mkPen("w", width=1)),
                ring=self.plot.plot([], [], pen=None, symbol="d", symbolSize=12, symbolBrush=colour,
                                    symbolPen=pg.mkPen("w", width=1)),
                lobe=self.plot.plot([], [], pen=pg.mkPen(colour, width=3, style=QtCore.Qt.PenStyle.DotLine)),
                label=pg.TextItem("", color=colour, anchor=(0.5, 1.3)))
            self.plot.addItem(self.items[name]["label"])
            self.checks[name] = QtWidgets.QCheckBox(f"{name} cut")
            self.checks[name].setStyleSheet(f"color: {colour}; font-weight: bold")
            self.checks[name].setChecked(True)
        self.show_fit = QtWidgets.QCheckBox("fitted Gaussian (dashed)")
        self.show_fit.setChecked(True)
        self.normalise = QtWidgets.QCheckBox("normalise to the peak")
        self.log_y = QtWidgets.QCheckBox("log y")
        self.half = QtWidgets.QSpinBox()
        self.half.setRange(2, 2000)
        self.half.setValue(40)
        self.half.setPrefix("± ")
        self.half.setSuffix(" px")
        self.half.setToolTip("Range of the cuts on either side of the fitted centre")
        # ring / peak: the central lobe inside the first Airy zero r0 is left out
        self.r0 = QtWidgets.QDoubleSpinBox()
        self.r0.setRange(0.0, 1000.0)
        self.r0.setDecimals(2)
        self.r0.setSuffix(" px")
        self.r0.setSpecialValueText("type r0")                  # 0 = unknown: no ratio
        self.r0.setValue(r0_px or 0.0)
        self.r0.setToolTip("First zero of the Airy pattern on the camera, r0 = 0.61 λ/NA × magnification / pixel "
                           "size, set from the imaging system (or typed). The central lobe |s| < r0 is fitted, "
                           "and ignored for the ring / peak ratio.")
        self.band = pg.LinearRegionItem(movable=False, brush=(255, 255, 255, 30), pen=pg.mkPen((255, 255, 255, 90)))
        self.band.setZValue(-10)
        self.plot.addItem(self.band)
        self.ratio = QtWidgets.QLabel()
        font = self.ratio.font()
        font.setPointSize(13)
        self.ratio.setFont(font)
        self.ratio.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.r0.valueChanged.connect(lambda _: self._redraw())
        for w in (*self.checks.values(), self.show_fit):
            w.toggled.connect(lambda _: self._redraw())
        for w in (self.normalise, self.log_y):
            w.toggled.connect(lambda _: (self._set_ranges(), self._redraw()))
        self.half.valueChanged.connect(lambda _: (self._set_ranges(), self._redraw()))

        controls = QtWidgets.QHBoxLayout()
        for w in (*self.checks.values(), self.show_fit, self.normalise, self.log_y):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("first zero r0"))
        controls.addWidget(self.r0)
        controls.addWidget(QtWidgets.QLabel("range"))
        controls.addWidget(self.half)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.ratio)
        # central-lobe fit values, fixed beside the plot
        self.side = QtWidgets.QLabel()
        self.side.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.side.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.side.setMinimumWidth(190)
        self.side.setWordWrap(True)
        font = self.side.font()
        font.setPointSize(12)
        self.side.setFont(font)
        title = QtWidgets.QLabel("<b>Central-lobe fit</b>")
        title.setFont(font)
        optics_button = QtWidgets.QPushButton("Imaging system…")
        optics_button.setToolTip("Wavelength, magnification, nominal NA and camera pixel: set r0 and the NA")
        optics_button.clicked.connect(self.edit_optics)
        self.optics_label = QtWidgets.QLabel()
        self.optics_label.setStyleSheet("color: gray")
        self.optics_label.setWordWrap(True)
        self.free_offset = QtWidgets.QCheckBox("free offset")
        self.free_offset.setToolTip("Fit the offset as well. Inside r0 alone it is poorly pinned and absorbs the "
                                    "non-Gaussian shape of the lobe (it can go tens of counts negative, w up); "
                                    "unticked, it is the background level of the cut's wings.")
        self.free_offset.toggled.connect(lambda _: self._redraw())
        # one row per cut: maximum of the cut, 1/e² radius of the lobe fit, and the NA from it
        self.table = QtWidgets.QTableWidget(len(COLOURS), 4)
        self.table.setHorizontalHeaderLabels(["max", "w (px)", "w (nm)", "NA"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFont(font)
        for i, name in enumerate(COLOURS):
            head = QtWidgets.QTableWidgetItem(f"{name} cut")
            head.setForeground(pg.mkColor(COLOURS[name]))
            self.table.setVerticalHeaderItem(i, head)
            for j in range(4):
                cell = QtWidgets.QTableWidgetItem("–")
                cell.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, cell)
        self.table.horizontalHeaderItem(0).setToolTip("Highest value of the cut (counts)")
        self.table.horizontalHeaderItem(1).setToolTip("1/e² radius of the Gaussian fitted to |s| ≤ r0, ± fit error")
        self.table.horizontalHeaderItem(2).setToolTip("The same in the object plane, pixel widening removed "
                                                      "(σ² − p²/12)")
        self.table.horizontalHeaderItem(3).setToolTip("NA = 0.421 λ / w (object plane), ± from the fit error of w "
                                                      "only; as the analytic NA of analyze_na.py")
        # fixed column widths for the widest values, so the live numbers never resize the table
        metrics = QtGui.QFontMetrics(font)
        for col, widest in enumerate(("8888", "88.88 ± 8.88", "8888 ± 888", "0.888 ± 0.888")):
            self.table.setColumnWidth(col, metrics.horizontalAdvance(widest) + 16)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setFixedWidth(metrics.horizontalAdvance("y cut") + 16)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setFixedWidth(self.table.verticalHeader().width() + 2 * self.table.frameWidth()
                                 + sum(self.table.columnWidth(c) for c in range(4)))
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height() + 2 * self.table.frameWidth()
                                  + sum(self.table.rowHeight(i) for i in range(len(COLOURS))))
        side = QtWidgets.QVBoxLayout()
        side.addWidget(title)
        side.addWidget(optics_button)
        side.addWidget(self.optics_label)
        side.addWidget(self.free_offset)
        side.addWidget(self.table)
        side.addWidget(self.side, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.plot, 1)
        row.addLayout(side)
        layout.addLayout(row, 1)
        self._set_ranges()
        self._show_optics()

    # ---------- imaging system ----------

    def edit_optics(self):
        dialog = OpticsDialog(self.optics, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.set_optics(dialog.values())

    def set_optics(self, optics: dict):
        """New imaging system: r0 follows it (if it can be computed), the NA column too."""
        self.optics = {**self.optics, **optics}
        if (r0 := first_zero_px(self.optics)) is not None:
            self.r0.setValue(r0)                                  # redraws
        self._show_optics()
        self._redraw()
        self.opticsChanged.emit(dict(self.optics))

    def _show_optics(self):
        o = self.optics
        missing = [OPTICS_FIELDS[k][0].lower() for k in OPTICS_FIELDS if not o.get(k)]
        if missing:
            self.optics_label.setText(f"Set the {', '.join(missing)} for the NA (and r0).")
            return
        px_nm = 1000.0 * o["pixel_size_um"] / o["magnification"]
        self.optics_label.setText(f"λ {o['wavelength_nm']:g} nm, M {o['magnification']:g}, pixel "
                                  f"{o['pixel_size_um']:g} µm → {px_nm:.1f} nm/px on the object; nominal NA "
                                  f"{o['na']:g} → r0 {first_zero_px(o):.2f} px")

    def _table_row(self, row: int, peak: float | None, lobe: dict | None):
        """One cut in the table: its maximum, the lobe fit's w and the NA from it ('–' where unknown)."""
        na = na_from_lobe(lobe["w"], lobe["w_err"], self.optics) if lobe is not None else None
        texts = ("–" if peak is None else f"{peak:.0f}",
                 "–" if lobe is None else f"{lobe['w']:.2f} ± {lobe['w_err']:.2f}",
                 "–" if na is None else f"{na['w_nm']:.0f} ± {na['w_err_nm']:.0f}",
                 "–" if na is None else f"{na['na']:.3f} ± {na['na_err']:.3f}")
        for col, text in enumerate(texts):
            self.table.item(row, col).setText(text)

    def _set_ranges(self):
        """Fixed axes: they only change with the settings (or the mouse), never with the data."""
        norm, log = self.normalise.isChecked(), self.log_y.isChecked()
        top = (1.0 if norm else self.max_value) * (2.0 if log else 1.15 if not norm else 1.25)   # room for the labels
        low = (1e-3 if norm else 0.5) if log else 0.0
        self.plot.setLogMode(y=log)
        self.plot.setLabel("left", "fraction of the peak" if norm else "counts")
        self.plot.setYRange(np.log10(low) if log else low, np.log10(top) if log else top, padding=0)
        self.plot.setXRange(-self.half.value(), self.half.value(), padding=0.02)

    def show_cuts(self, frame: np.ndarray | None, fit: dict | None, offset: tuple[int, int] = (0, 0)):
        """
        New fit (centre in the array coordinates of `frame`; offset = (x, y) of the
        frame on the sensor, for the row / column numbers); None, None after a failed fit.
        """
        self._last = None if fit is None else (frame, fit)
        self._offset = offset
        self._redraw()

    def _redraw(self):
        cuts = centre_cuts(*self._last, self.half.value()) if self._last is not None else {}
        ox, oy = getattr(self, "_offset", (0, 0))
        log, titles, ratios, lobes, r0 = self.log_y.isChecked(), [], [], [], self.r0.value()
        self.band.setRegion((-r0, r0))
        self.band.setVisible(r0 > 0)
        for row, (name, parts) in enumerate(self.items.items()):
            on = self.checks[name].isChecked()
            for item in parts.values():
                item.setVisible(on)
            parts["gaussian"].setVisible(on and self.show_fit.isChecked())
            cut = cuts.get(name)
            if cut is None or not np.isfinite(cut["values"]).any():
                for key in ("data", "gaussian", "marker", "ring", "lobe"):
                    parts[key].setData([], [])
                parts["label"].setText("")
                self._table_row(row, None, None)
                continue
            i = int(np.nanargmax(cut["values"]))
            peak = cut["values"][i]
            scale = 1.0 / peak if self.normalise.isChecked() and peak > 0 else 1.0
            values, gaussian = cut["values"] * scale, cut["gaussian"] * scale
            if log:                                   # nothing at or below zero on a log axis
                values, gaussian = (np.where(v > 0, v, np.nan) for v in (values, gaussian))
            parts["data"].setData(cut["s"], values)
            parts["gaussian"].setData(cut["s_fine"], gaussian)
            parts["marker"].setData([cut["s"][i]], [peak * scale])
            parts["label"].setText(f"{peak:.0f}")
            y = peak * scale
            parts["label"].setPos(cut["s"][i], np.log10(y) if log and y > 0 else y)
            titles.append(f"{name} cut: {'row y' if name == 'x' else 'column x'} = "
                          f"{cut['line'] + (oy if name == 'x' else ox)}, fit w = {cut['w']:.2f} px")

            # Gaussian with offset fitted to the central lobe only (|s| <= r0); the offset is the background
            # level of the wings (|s| >= 4 r0; the 2D fit's offset if the range is shorter) unless left free
            wings = cut["values"][(np.abs(cut["s"]) >= 4 * r0) & np.isfinite(cut["values"])]
            level = None if self.free_offset.isChecked() else float(np.median(wings)) if wings.size else cut["offset"]
            lobe = lobe_fit(cut["s"], cut["values"], r0, cut["w"], level) if r0 > 0 else None
            self._table_row(row, peak, lobe)
            if lobe is None:
                parts["lobe"].setData([], [])
            else:
                s_lobe = np.linspace(-r0, r0, 101)
                curve = gaussian_1d(s_lobe, lobe["amplitude"], lobe["s0"], lobe["w"], lobe["offset"]) * scale
                parts["lobe"].setData(s_lobe, np.where(curve > 0, curve, np.nan) if log else curve)
                if on:
                    lobes.append(f"{name} {lobe['offset']:.1f}")

            # highest point outside the central lobe, relative to the peak, fitted offset removed from both
            outside = np.flatnonzero((np.abs(cut["s"]) >= r0) & np.isfinite(cut["values"]))
            if r0 <= 0 or outside.size == 0 or peak <= cut["offset"]:
                parts["ring"].setData([], [])
                continue
            j = outside[int(np.argmax(cut["values"][outside]))]
            ring = (cut["values"][j] - cut["offset"]) / (peak - cut["offset"])
            y = cut["values"][j] * scale
            parts["ring"].setData([cut["s"][j]], [y if y > 0 or not log else np.nan])
            if on:
                ratios.append(f"<b style='color: {COLOURS[name]}'>{name}&nbsp;{100 * ring:.1f}&nbsp;%</b>")
        self.plot.setTitle("   |   ".join(titles) if titles else "no Gaussian fit")
        how = "offset fitted too" if self.free_offset.isChecked() else "offset = wing median (|s| ≥ 4 r0)"
        self.side.setText(f"<span style='color: gray'>Gaussian + offset fitted to |s| ≤ r0, {how}"
                          + (f": {', '.join(lobes)} counts" if lobes else "") + "</span>"
                          + ("<br><br>Set the imaging system or type r0." if r0 <= 0 else ""))
        if r0 <= 0:
            self.ratio.setText("Type the first zero r0 to get the ring / peak ratio.")
        elif ratios:
            self.ratio.setText(f"Highest point outside r0 / peak:&nbsp;&nbsp;{'&nbsp;&nbsp;&nbsp;'.join(ratios)}"
                               f"&nbsp;&nbsp;&nbsp;<span style='color: gray'>(perfect Airy pattern "
                               f"{100 * AIRY_FIRST_RING:.2f} %; fitted offset removed)</span>")
        else:
            self.ratio.setText("No point outside r0 within the range: widen the range.")
