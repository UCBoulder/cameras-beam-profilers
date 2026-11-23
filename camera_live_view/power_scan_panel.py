"""
The power-meter scan in the live viewer's Alignment dock (power_scan.py does
the reading, fitting and saving).

Connect the meter, choose x or y, start, and move the beam slowly across the
aperture / fibre in front of the meter: every 2D Gaussian fit of the viewer
adds a point (spot centre, power read during that frame) to the live plot,
and a Gaussian on an offset is refitted to the points as they come in. Stop
fits once more, saves the scan (only then) and remembers the maximum of that
axis; once both axes have one, the viewer asks whether to mark (x0, y0) as a
reference point. While scanning, "Reset data" drops the points taken so far,
"Reset fit" forgets the fit of the axis on the plot (the next one starts from
fresh start values) and its remembered maximum.

LiveView builds the panel, passes it every successful fit (on_fit(), sensor
pixels) and calls shutdown() when it closes. From the viewer it uses: cam
(exposure_us, gain_db), data_root, _settings, _aoi, _region(), _raw_last,
fit_on, fit_window, fit_in_region and add_reference_point(name, x, y, note).
"""

import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .power_scan import PowerReader, fit_gaussian_offset, format_power, gaussian_offset, save_power_scan


class PowerScanPanel(QtWidgets.QGroupBox):

    LIVE_FIT_S = 0.5        # refit the points at most this often while scanning
    POWER_MS = 200          # live power readout refresh
    STALE_S = 3.0           # no new reading for this long: say so

    def __init__(self, viewer, power_meter=None):
        super().__init__("Power-meter scan (maximum transmission, x / y)")
        self.viewer = viewer
        self.meter, self.reader = None, None
        self._run = None            # running scan: dict(axis, t0, samples, paused)
        self._shown, self._shown_axis = [], "x"   # samples on the plot (running or last scan) and their axis
        self._fit, self._fit_time = None, 0.0     # Gaussian of the points on the plot
        self.peaks = {}             # axis -> maximum of the last finished scan along it (s0, s0_err, folder, ...)
        store = QtCore.QSettings(*viewer._settings)

        help_text = QtWidgets.QLabel(
            "Where on the camera is the power behind an aperture / fibre highest? Connect the power meter, choose "
            "x or y, start, and move the beam slowly across: every Gaussian fit of the spot adds a point (centre, "
            "power during that frame). Stop fits P = A exp(−2 (x − x0)²/w²) + B and saves "
            "&lt;folder&gt;/&lt;name&gt;_&lt;axis&gt;_&lt;time&gt;/ (scan.json, power_scan.csv, plot). After an x and "
            "a y scan, (x0, y0) can be marked as a reference point.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray")

        # power meter
        self.meter_list = QtWidgets.QComboBox()
        self.meter_list.setToolTip("Power meters found (Thorlabs, TLPM driver); 'Find' searches again")
        self.meter_list.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.meter_list.setMinimumContentsLength(14)   # fixed width ('PM100D P0031248'); the list shows longer names
        find = QtWidgets.QPushButton("Find")
        find.clicked.connect(lambda: self.find_meters())
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(lambda: self.toggle_connect())
        meter_row = QtWidgets.QHBoxLayout()
        meter_row.addWidget(self.meter_list)
        meter_row.addWidget(find)
        meter_row.addWidget(self.btn_connect)
        meter_row.addStretch(1)            # the buttons next to the meter name, not at the far edge
        self.meter_label = QtWidgets.QLabel("not connected")
        self.meter_label.setWordWrap(True)
        self.meter_label.setStyleSheet("color: gray")
        self.wavelength = QtWidgets.QDoubleSpinBox()
        self.wavelength.setRange(100.0, 20000.0)
        self.wavelength.setDecimals(1)
        self.wavelength.setSuffix(" nm")
        self.wavelength.setKeyboardTracking(False)
        self.wavelength.setValue(float(store.value("pm_wavelength_nm", viewer.optics.get("wavelength_nm") or 520.0)))
        self.wavelength.setToolTip("Wavelength of the sensor's responsivity correction (absolute power only; the "
                                   "position of the maximum does not depend on it)")
        self.wavelength.valueChanged.connect(self._set_wavelength)
        self.power_label = QtWidgets.QLabel("—")
        font = self.power_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.power_label.setFont(font)

        # scan
        self.name = QtWidgets.QLineEdit(str(store.value("pm_name", "power scan")))
        self.folder = QtWidgets.QLineEdit(str(store.value("pm_folder", viewer.data_root / "alignment_data")))
        self.folder.setToolTip("Each scan is saved in a new subfolder <name>_<axis>_<time> of this folder")
        browse = QtWidgets.QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(lambda: self._browse())
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        self.axis_x, self.axis_y = QtWidgets.QRadioButton("x scan"), QtWidgets.QRadioButton("y scan")
        self.axis_x.setChecked(True)
        self.axis_x.toggled.connect(lambda _: self._buttons())
        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(self.axis_x)
        axis_row.addWidget(self.axis_y)
        axis_row.addStretch(1)
        form = QtWidgets.QFormLayout()
        form.addRow("Meter", meter_row)
        form.addRow(self.meter_label)
        form.addRow("Wavelength", self.wavelength)
        form.addRow("Power", self.power_label)
        form.addRow("Name", self.name)
        form.addRow("Save in", folder_row)
        form.addRow("Axis", axis_row)
        self._browse_btn = browse

        self.btn_start = QtWidgets.QPushButton("Start x scan")
        self.btn_start.clicked.connect(lambda: self.start())
        self.btn_stop = QtWidgets.QPushButton("Stop, fit and save")
        self.btn_stop.clicked.connect(lambda: self.stop())
        self.btn_reset_data = QtWidgets.QPushButton("Reset data")
        self.btn_reset_data.setToolTip("Drop the points taken so far (the scan keeps running)")
        self.btn_reset_data.clicked.connect(lambda: self.reset_data())
        self.btn_reset_fit = QtWidgets.QPushButton("Reset fit")
        self.btn_reset_fit.setToolTip("Forget the fit of the axis on the plot and its remembered maximum; "
                                      "the next fit starts from fresh start values")
        self.btn_reset_fit.clicked.connect(lambda: self.reset_fit())
        resets = QtWidgets.QHBoxLayout()
        resets.addWidget(self.btn_reset_data)
        resets.addWidget(self.btn_reset_fit)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # auto-range margin: pyqtgraph pads by 50 × this / √(plot size in px), so ≈ 2 % here instead of ≈ 7 %
        self.plot.getViewBox().setDefaultPadding(0.006)
        self.plot.setMouseEnabled(x=False, y=False)     # the axes are always the automatic ones: no zoom / pan,
        self.plot.setMenuEnabled(False)                 # no right-click menu, no auto-range button
        self.plot.hideButtons()
        self.plot.setLabel("left", "power", units="W")
        self.plot.setLabel("bottom", "x (sensor px)")
        self.points = self.plot.plot([], [], pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                     symbolBrush=(100, 180, 255))
        # The axes scale to the measured points (x: all of them, y: lowest to highest power) and to the fitted
        # Gaussian only through _fit_extent (invisible), which holds it while its maximum is a sensible one: a
        # runaway fit (one flank scanned, peak far outside at kW) never rescales the plot.
        self.curve = pg.PlotDataItem(pen=pg.mkPen((255, 170, 0), width=2))
        self.plot.addItem(self.curve, ignoreBounds=True)
        self._fit_extent = pg.PlotDataItem(pen=pg.mkPen(None))
        self.plot.addItem(self._fit_extent)
        self.peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 60, 60), width=2,
                                                                style=QtCore.Qt.PenStyle.DashLine))
        self.peak_line.hide()
        self.plot.addItem(self.peak_line, ignoreBounds=True)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.btn_mark = QtWidgets.QPushButton("Mark the x and y maxima as a reference point…")
        self.btn_mark.clicked.connect(lambda: self.offer_mark())

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(help_text)
        layout.addLayout(form)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addLayout(resets)
        layout.addWidget(self.plot)
        layout.addWidget(self.status)
        layout.addWidget(self.btn_mark)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.POWER_MS)
        self.timer.timeout.connect(self._show_power)
        if power_meter is not None:
            self._connect(power_meter)
        self._buttons()
        self._say()

    # ---------- power meter ----------

    def find_meters(self) -> bool:
        """List the Thorlabs meters in the drop-down; False (and the reason shown) if none."""
        self.meter_list.clear()
        try:
            from thorlabs_power_meter import find_power_meters
            found = find_power_meters()
        except Exception as err:            # driver not installed, ...
            self.meter_label.setText(f"<span style='color: #c00000'>{err}</span>")
            return False
        for resource, label in found:
            self.meter_list.addItem(label, resource)
        if not found:
            self.meter_label.setText("No power meter found: plugged in? Close Thorlabs' Optical Power Monitor; "
                                     "switch the meter to the TLPM driver if it is on another one.")
        return bool(found)

    def toggle_connect(self):
        if self.meter is not None:
            self.disconnect_meter()
            return
        if self.meter_list.count() == 0 and not self.find_meters():
            return
        try:
            from thorlabs_power_meter import ThorlabsPowerMeter
            meter = ThorlabsPowerMeter(self.meter_list.currentData(), wavelength_nm=self.wavelength.value())
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Power meter", f"{type(err).__name__}: {err}")
            return
        self._connect(meter)

    def _connect(self, meter):
        self.meter, self.reader = meter, PowerReader(meter)
        self.meter_label.setText(getattr(meter, "description", type(meter).__name__))
        if hasattr(meter, "wavelength_nm"):
            try:
                wavelength = self.reader.call(lambda: meter.wavelength_nm)
                self.wavelength.blockSignals(True)
                self.wavelength.setValue(wavelength)
                self.wavelength.blockSignals(False)
            except Exception:
                pass
        self.timer.start()
        self._buttons()

    def disconnect_meter(self):
        if self._run is not None:
            QtWidgets.QMessageBox.information(self, "Power meter", "Stop the scan first.")
            return
        self.timer.stop()
        self.reader.stop()
        try:
            self.meter.close()
        except Exception as err:
            print(f"[power scan] closing the meter: {err}")
        self.meter = self.reader = None
        self.meter_label.setText("not connected")
        self.power_label.setText("—")
        self._buttons()

    def _set_wavelength(self, nm: float):
        if self.meter is not None and hasattr(self.meter, "set_wavelength_nm"):
            try:
                self.reader.call(self.meter.set_wavelength_nm, nm)
            except Exception as err:
                QtWidgets.QMessageBox.warning(self, "Power meter", f"Wavelength not set: {err}")

    def _show_power(self):
        latest = self.reader.latest() if self.reader else None
        if self.reader and self.reader.error:
            self.power_label.setText(f"<span style='color: #c00000; font-size: 9pt'>{self.reader.error}</span>")
        elif latest is None:
            self.power_label.setText("waiting…")
        else:
            age = time.perf_counter() - latest[0]
            self.power_label.setText(format_power(latest[1]) + (f"  <span style='color: #c00000; font-size: 9pt'>"
                                                                f"(no reading for {age:.0f} s)</span>"
                                                                if age > self.STALE_S else ""))

    def _meter_info(self) -> dict:
        try:
            if hasattr(self.meter, "info"):
                return self.reader.call(self.meter.info)
        except Exception as err:
            return dict(description=getattr(self.meter, "description", ""), info_error=str(err))
        return dict(description=getattr(self.meter, "description", type(self.meter).__name__))

    # ---------- scan ----------

    @property
    def axis(self) -> str:
        return "x" if self.axis_x.isChecked() else "y"

    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save power-meter scans in", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def start(self):
        if self.meter is None:
            QtWidgets.QMessageBox.information(self, "Power-meter scan", "Connect the power meter first.")
            return
        if self.viewer._raw_last is None:
            QtWidgets.QMessageBox.information(self, "Power-meter scan", "No frame yet: start the live stream.")
            return
        try:
            Path(self.folder.text()).mkdir(parents=True, exist_ok=True)    # fail now, not after the scan
        except OSError as err:
            QtWidgets.QMessageBox.critical(self, "Power-meter scan", f"Cannot use that folder: {err}")
            return
        self.viewer.fit_on.setChecked(True)
        self._run = dict(axis=self.axis, t0=time.perf_counter(), samples=[], paused=False)
        self._shown, self._shown_axis, self._fit = self._run["samples"], self.axis, None
        self._redraw()
        self._buttons()
        self._say()

    def on_fit(self, fit: dict):
        """A new 2D Gaussian fit (sensor pixels): a point of the running scan, with the mean power read
        during that frame's exposure."""
        run = self._run
        if run is None or run["paused"] or self.reader is None:
            return
        now = time.perf_counter()
        power, n = self.reader.mean_since(now - max(1e-6 * self.viewer.cam.exposure_us, 0.05))
        if power is None:
            return
        run["samples"].append(dict(t_s=round(now - run["t0"], 4), x_px=float(fit["x0"]), y_px=float(fit["y0"]),
                                   power_w=power, n_readings=n, fit_peak_counts=float(fit["peak"]),
                                   wx_px=float(fit["wx"]), wy_px=float(fit["wy"])))
        if now - self._fit_time >= self.LIVE_FIT_S:
            self._refit()
            self._fit_time = now
        self._redraw()
        self._buttons()                     # Reset fit / data become available
        self._say()

    def _refit(self):
        """Fit the points on the plot, from the current fit's values (steadier while points come in), else
        from fresh start values."""
        s, p = self._arrays()
        self._fit = (fit_gaussian_offset(s, p, start=self._fit) if self._fit else None) or fit_gaussian_offset(s, p)

    def stop(self):
        """Fit, save, remember the maximum, offer the reference point once x and y have one."""
        run = self._run
        if run is None:
            return
        run["paused"] = True
        axis, samples = run["axis"], run["samples"]
        if not samples:
            self._run = None
            self._buttons()
            self._say("Stopped: no points recorded, nothing saved.")
            return
        self._refit()
        v = self.viewer
        try:
            folder = save_power_scan(self.folder.text(), self.name.text().strip() or "power scan", axis, samples,
                                     self._fit, camera=v.cam, meter_info=self._meter_info(),
                                     exposure_us=v.cam.exposure_us, gain_db=v.cam.gain_db, aoi=dict(v._aoi),
                                     fit_window=v.fit_window.currentText(),
                                     spot_in_analysis_region=v.fit_in_region.isChecked(),
                                     analysis_region=v._region())
        except Exception as err:            # the points stay: another folder, then Stop again
            QtWidgets.QMessageBox.critical(self, "Power-meter scan", f"Not saved ({type(err).__name__}: {err}). "
                                           "The points are kept: choose another folder and press Stop again.")
            self._buttons()
            return
        self._run = None
        other = "y" if axis == "x" else "x"
        if self._fit:
            self.peaks[axis] = dict(self._fit, folder=str(folder),
                                    other=float(np.mean([r[f"{other}_px"] for r in samples])))
        self._redraw()
        if other not in self.peaks:                                     # the other axis next
            (self.axis_y if other == "y" else self.axis_x).setChecked(True)
        self._buttons()
        self._say(f"Saved {len(samples)} points to {folder}"
                  + ("" if self._fit else "<br><span style='color: #c00000'>The Gaussian fit failed: no maximum "
                                         "remembered for this axis.</span>"))
        if self._fit and "x" in self.peaks and "y" in self.peaks:
            self.offer_mark()

    def reset_data(self):
        """Drop the points (of the running scan, or the last one on the plot) and their fit."""
        if self._run is not None:
            self._run["samples"].clear()
        else:
            self._shown = []
        self._fit = None
        self._redraw()
        self._buttons()
        self._say("Points dropped." if self._run is None else "")

    def reset_fit(self):
        self._fit = None
        self._fit_time = time.perf_counter()     # while scanning, the next refit waits LIVE_FIT_S
        forgotten = self.peaks.pop(self._shown_axis, None)
        self._redraw()
        self._buttons()
        self._say(f"Fit reset; the {self._shown_axis} maximum is forgotten." if forgotten else "Fit reset.")

    def offer_mark(self):
        """Ask whether to mark (x0, y0) of the two last scans as a reference point."""
        if not ("x" in self.peaks and "y" in self.peaks):
            return
        px, py = self.peaks["x"], self.peaks["y"]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Power-meter maxima",
            f"Maximum power at x = {px['s0']:.2f} ± {px['s0_err']:.2f} px, y = {py['s0']:.2f} ± {py['s0_err']:.2f} px "
            f"(sensor pixels, top-left pixel 0).\nThe x scan ran at y ≈ {px['other']:.1f} px, the y scan at "
            f"x ≈ {py['other']:.1f} px.\n\nMark it as a reference point named:", text="power maximum")
        if ok and name.strip():
            self.viewer.add_reference_point(name.strip(), px["s0"], py["s0"], note="power-meter maxima")

    # ---------- display ----------

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.array([r[f"{self._shown_axis}_px"] for r in self._shown], float)
        return s, np.array([r["power_w"] for r in self._shown], float)

    def _redraw(self):
        s, p = self._arrays()
        self.points.setData(s, p)
        self.plot.setLabel("bottom", f"{self._shown_axis} (sensor px)")
        fit = self._fit
        if fit and len(s):
            grid = np.linspace(s.min(), s.max(), 300)    # the scanned range, which is what the plot shows
            curve = gaussian_offset(grid, fit["amplitude"], fit["s0"], fit["w"], fit["offset"])
            self.curve.setData(grid, curve)
            # the whole Gaussian in view when its maximum lies in the scanned range and at most half the measured
            # power range above the highest point; otherwise the axes follow the points alone
            if fit["inside"] and curve.max() <= p.max() + np.ptp(p) / 2:
                self._fit_extent.setData(grid, curve)
            else:
                self._fit_extent.setData([], [])
            self.peak_line.setPos(fit["s0"])
            self.peak_line.show()
        else:
            self.curve.setData([], [])
            self._fit_extent.setData([], [])
            self.peak_line.hide()

    def _say(self, text: str = ""):
        lines = [text] if text else []
        if self._run is not None:
            lines.append(f"Recording the <b>{self._run['axis']} scan</b>: {len(self._run['samples'])} points. "
                         f"Move the beam slowly across; stop to fit and save.")
        fit = self._fit
        if fit and self._shown:
            lines.append(f"Fit: maximum at {self._shown_axis} = <b>{fit['s0']:.2f}</b> ± {fit['s0_err']:.2f} px, "
                         f"w = {fit['w']:.1f} px, peak {format_power(fit['peak_power'])}"
                         + ("" if fit["inside"] else " <span style='color: #c00000'>(outside the scanned range: "
                                                     "scan further)</span>"))
        for axis in ("x", "y"):
            if axis in self.peaks:
                lines.append(f"Remembered {axis} maximum: <b>{self.peaks[axis]['s0']:.2f}</b> ± "
                             f"{self.peaks[axis]['s0_err']:.2f} px ({Path(self.peaks[axis]['folder']).name})")
        self.status.setText("<br>".join(lines))

    def _buttons(self):
        connected, running = self.meter is not None, self._run is not None
        self.btn_connect.setText("Disconnect" if connected else "Connect")
        self.btn_connect.setEnabled(not running)
        self.meter_list.setEnabled(not connected)
        self.wavelength.setEnabled(connected and hasattr(self.meter, "set_wavelength_nm"))
        self.btn_start.setText(f"Start {self.axis} scan")
        self.btn_start.setEnabled(connected and not running)
        self.btn_stop.setEnabled(running)
        for w in (self.axis_x, self.axis_y, self.name, self.folder, self._browse_btn):
            w.setEnabled(not running)
        self.btn_reset_data.setEnabled(bool(self._shown) or running)
        self.btn_reset_fit.setEnabled(self._fit is not None or self._shown_axis in self.peaks)
        self.btn_mark.setEnabled("x" in self.peaks and "y" in self.peaks)

    def shutdown(self):
        """Save the fields, stop reading and release the meter (viewer closing)."""
        store = QtCore.QSettings(*self.viewer._settings)
        store.setValue("pm_name", self.name.text())
        store.setValue("pm_folder", self.folder.text())
        store.setValue("pm_wavelength_nm", self.wavelength.value())
        self._run = None
        if self.meter is not None:
            self.disconnect_meter()
