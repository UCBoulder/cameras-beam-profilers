"""
Live camera view: the image on a logarithmic colour scale, exposure and gain
controls, zoom / pan, a region of interest that can be sent to the camera,
and a snapshot button.

    from cinogy_uc480 import CinCamUC480
    from camera_live_view import run_live_view

    cam = CinCamUC480({"data_root": "data", "exposure_us": 8000})
    try:
        run_live_view(cam, data_root="data", experiment="alignment")
    finally:
        cam.close()

Works with any camera that has the live interface of cinogy_uc480.CinCamUC480
(currently the only one): start_live(), stop_live(), live, grab_latest(),
frame_rate(); set_exposure(), set_gain(), exposure_us, gain_db,
exposure_range_us(), gain_range_db(); set_aoi(), reset_aoi(), aoi,
sensor_width, sensor_height, max_value; snapshot() and data_root.

Data flow
---------
The camera runs free (start_live) into its own buffers. A QTimer in the GUI
thread polls grab_latest() about 30 times a second and draws the newest
frame. That call never blocks and skips frames the GUI did not get to, so
the display always shows the current state and stays responsive without any
threads. Settings go straight to the camera (they take effect within two
frames). A snapshot uses the camera's own snapshot(), which pauses the
stream for it, so the files are exactly those of cam.snapshot().

A recorded background (one snapshot, saved under data/background/ by
default, or any data_*.npy loaded from disk) can be subtracted from the live
feed; snapshots always save the raw frame.

With "Gaussian fit" on, a rotated 2D Gaussian is fitted to the displayed
spot FIT_PER_SECOND times a second (gaussian_fit.py, ~10 ms per fit at any
window size thanks to binning): the
centre and 1/e² radii are shown under the image, in pixels or µm on the
sensor, from the top-left pixel or the camera centre, and the ellipse is
drawn on the image.

Layout after Allied Vision's viewer: toolbar on top (live on/off, save),
controls in a dock on the left, image in the middle. Mouse wheel zooms,
right-drag (or left-drag outside the rectangle) pans; the yellow rectangle is
the camera's readout region (drag / resize it, then "Apply to camera").
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .gaussian_fit import fit_gaussian_2d
from .dead_pixels import all_bad_pixels, find_dead_pixels, load_dead_pixels, repair, save_dead_pixels

# Beam-profiler colour scale, the same as PSFPlotter.BEAM_CMAP: dark blue ->
# blue -> cyan -> green -> yellow -> red, white only at saturation
_BASE_STOPS = [0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0]
_BASE_COLOURS = [(0, 0, 77), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 255, 255)]


def colour_map(max_value: int, log: bool = True) -> pg.ColorMap:
    """The base colours over 0 .. max_value, counts mapped logarithmically if log."""
    base = pg.ColorMap(_BASE_STOPS, _BASE_COLOURS)
    u = np.linspace(0.0, 1.0, 256)
    pos = np.log1p(u * max_value) / np.log1p(max_value) if log else u
    return pg.ColorMap(u, base.map(pos, mode=pg.ColorMap.BYTE))


class _PanViewBox(pg.ViewBox):
    """ViewBox where a right-button drag pans (left drag on the ROI moves the ROI instead)."""

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            to_view = pg.functions.invertQTransform(self.childGroup.transform())
            shift = to_view.map(ev.lastPos()) - to_view.map(ev.pos())
            self.translateBy(x=shift.x(), y=shift.y())
            self.sigRangeChangedManually.emit(self.state["mouseEnabled"])
        else:
            super().mouseDragEvent(ev, axis)


class _ParamControl(QtWidgets.QWidget):
    """
    Spin box + slider for one camera parameter. valueChanged fires when the
    user is done (slider released, value typed), not on every pixel of a drag.
    """

    valueChanged = QtCore.Signal(float)
    _STEPS = 1000

    def __init__(self, lo: float, hi: float, decimals: int, suffix: str, log: bool = False):
        super().__init__()
        self.lo, self.hi, self.log = lo, hi, log
        self.spin = QtWidgets.QDoubleSpinBox(decimals=decimals, suffix=suffix)
        self.spin.setRange(lo, hi)
        self.spin.setKeyboardTracking(False)  # emit on Enter / focus loss, not per keystroke
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._STEPS)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider)

        self.spin.valueChanged.connect(self.valueChanged)
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderReleased.connect(lambda: self.valueChanged.emit(self._from_slider(self.slider.value())))

    def _slider_changed(self, step: int):
        value = self._from_slider(step)
        self._show(value)
        if not self.slider.isSliderDown():  # click or keyboard: no release coming
            self.valueChanged.emit(value)

    def _show(self, value: float):
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)

    def setValue(self, value: float):
        """Show a value without emitting valueChanged."""
        self._show(value)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)

    def _to_slider(self, value: float) -> int:
        if self.log:
            f = math.log(max(value, self.lo) / self.lo) / math.log(self.hi / self.lo)
        else:
            f = (value - self.lo) / (self.hi - self.lo)
        return int(round(self._STEPS * min(max(f, 0.0), 1.0)))

    def _from_slider(self, step: int) -> float:
        f = step / self._STEPS
        return self.lo * (self.hi / self.lo) ** f if self.log else self.lo + f * (self.hi - self.lo)


class SnapshotDialog(QtWidgets.QDialog):
    """Experiment name, data folder and comment for a snapshot."""

    def __init__(self, parent, experiment: str, data_root: Path):
        super().__init__(parent)
        self.setWindowTitle("Save snapshot")
        self.experiment = QtWidgets.QComboBox()   # type a new name or pick an existing folder
        self.experiment.setEditable(True)
        self.root = QtWidgets.QLineEdit(str(data_root))
        self.comment = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        self.target = QtWidgets.QLabel()
        self.target.setStyleSheet("color: gray")
        note = QtWidgets.QLabel("The raw frame is saved: background subtraction only affects the display.\n"
                                "The background file in use is recorded in the metadata.")
        note.setStyleSheet("color: gray")
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(self.root)
        root_row.addWidget(browse)
        form = QtWidgets.QFormLayout(self)
        form.addRow("Experiment", self.experiment)
        form.addRow("Data folder", root_row)
        form.addRow("Comment", self.comment)
        form.addRow(self.target)
        form.addRow(note)
        form.addRow(self.buttons)
        self.experiment.lineEdit().textChanged.connect(self._update_target)
        self.root.textChanged.connect(self._fill_experiments)
        self._fill_experiments()
        self.experiment.setEditText(experiment)
        self.resize(560, self.sizeHint().height())

    def _fill_experiments(self):
        """Drop-down: the experiment folders already in the data folder."""
        current = self.experiment.currentText()
        root = Path(self.root.text())
        self.experiment.clear()
        if root.is_dir():
            self.experiment.addItems(sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")))
        self.experiment.setEditText(current)
        self._update_target()

    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", self.root.text())
        if folder:
            self.root.setText(folder)

    def _update_target(self):
        name = self.experiment.currentText().strip()
        self.target.setText(f"→ {Path(self.root.text()) / name / 'snapshot_<timestamp>'}")
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(bool(name))

    def values(self) -> tuple[str, Path, str]:
        return self.experiment.currentText().strip(), Path(self.root.text()), self.comment.text()


class LiveView(QtWidgets.QMainWindow):

    POLL_MS = 30       # grab_latest() polling period
    PANEL_MIN_WIDTH = 270   # settings column (px); drag the separator to widen it
    STATUS_S = 0.5     # status-bar refresh
    FIT_PER_SECOND = 10.0   # at most; a fit slower than the interval just delays the next one
    DEAD_PIXEL_FRAMES = 8   # frames averaged by "Find dead pixels"

    def __init__(self, cam, data_root: str | Path, experiment: str = "live view"):
        super().__init__()
        self.cam = cam
        self.data_root = Path(data_root)
        self.experiment = experiment
        self.setWindowTitle(f"Live view — {type(cam).__name__}")
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._shape = None          # shape of the frame on screen; changes with the AOI
        self._aoi = cam.aoi
        self._last = None
        self._background = None     # int32 array and the readout rectangle it was taken with
        self._background_aoi = None
        self._background_path = None
        self._background_settings = None   # (exposure_us, gain_db) the background was taken with
        self._bg_slice = None       # part of the background covering the current readout
        self._frames, self._t_status = 0, time.perf_counter()
        self._dead = None           # dead pixels and dark spots (dead_pixels.py dict), sensor pixels
        self._dead_path = Path(getattr(cam, "dead_pixel_file", self.data_root / "dead_pixels.json"))
        self._dead_accum = None     # raw frames being averaged for "Find dead pixels"

        self._build_toolbar()
        self._build_controls()
        self._build_image()
        self.status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.POLL_MS)
        self.timer.timeout.connect(self._poll)
        self.act_live.setChecked(True)

    # ---------- Widgets ----------

    def _icon(self, name: str) -> QtGui.QIcon:
        return self.style().standardIcon(getattr(QtWidgets.QStyle.StandardPixmap, name))

    def _build_toolbar(self):
        bar = self.addToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(24, 24))
        self.act_live = QtGui.QAction(self._icon("SP_MediaPlay"), "Live", self)
        self.act_live.setCheckable(True)
        self.act_live.toggled.connect(self.set_live)
        self.act_save = QtGui.QAction(self._icon("SP_DialogSaveButton"), "Save snapshot…", self)
        self.act_save.triggered.connect(self.save_snapshot)
        bar.addAction(self.act_live)
        bar.addAction(self.act_save)
        self.toolbar = bar

    @staticmethod
    def _group(title: str, *widgets) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        for w in widgets:
            layout.addWidget(w)
        return box

    def _build_controls(self):
        lo, hi = self.cam.exposure_range_us()
        self.exposure = _ParamControl(lo, hi, 1, " µs", log=True)
        self.exposure.setValue(self.cam.exposure_us)
        self.exposure.valueChanged.connect(self._set_exposure)

        glo, ghi = self.cam.gain_range_db()
        self.gain = _ParamControl(glo, ghi if ghi > glo else glo + 1, 1, " dB")
        self.gain.setEnabled(ghi > glo)
        self.gain.setValue(self.cam.gain_db)
        self.gain.valueChanged.connect(self._set_gain)

        # ROI: x, y, width, height spin boxes kept in step with the rectangle on the image
        self.roi_spins = []
        roi_grid = QtWidgets.QGridLayout()
        for i, (label, maximum) in enumerate((("X", self.cam.sensor_width), ("Y", self.cam.sensor_height),
                                              ("W", self.cam.sensor_width), ("H", self.cam.sensor_height))):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, maximum)
            spin.setSuffix(" px")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._spins_changed)
            roi_grid.addWidget(QtWidgets.QLabel(label), i // 2, 2 * (i % 2))
            roi_grid.addWidget(spin, i // 2, 2 * (i % 2) + 1)
            self.roi_spins.append(spin)
        roi_widget = QtWidgets.QWidget()
        roi_widget.setLayout(roi_grid)
        apply = QtWidgets.QPushButton("Apply to camera")
        apply.clicked.connect(self._apply_roi)
        full = QtWidgets.QPushButton("Full frame")
        full.clicked.connect(self._full_frame)
        self.show_roi = QtWidgets.QCheckBox("Show rectangle")
        self.show_roi.setChecked(True)
        self.show_roi.toggled.connect(lambda on: self.roi.setVisible(on))

        self.bg_label = QtWidgets.QLabel("none")
        self.bg_label.setStyleSheet("color: gray")
        record = QtWidgets.QPushButton("Record background…")
        record.clicked.connect(lambda: self.record_background())  # clicked passes a bool
        load = QtWidgets.QPushButton("Load background…")
        load.clicked.connect(lambda: self.load_background())
        self.subtract = QtWidgets.QCheckBox("Subtract background")
        self.subtract.setEnabled(False)
        self.bg_warning = QtWidgets.QLabel()
        self.bg_warning.setWordWrap(True)
        self.bg_warning.setStyleSheet("color: red")
        self.bg_warning.hide()

        # Gaussian fit: on / off, units and origin of the readout, window around the spot
        self.fit_on = QtWidgets.QCheckBox(f"Fit a 2D Gaussian ({self.FIT_PER_SECOND:g} / s)")
        self.fit_on.toggled.connect(self._show_fit)
        self.fit_units = QtWidgets.QComboBox()
        self.fit_units.addItem("pixels", 1.0)
        pixel_um = getattr(self.cam, "pixel_size_um", None)
        if pixel_um:
            self.fit_units.addItem("µm on the sensor", float(pixel_um))
        self.fit_origin = QtWidgets.QComboBox()
        self.fit_origin.addItems(["top-left pixel", "camera centre"])
        self.fit_window = QtWidgets.QSpinBox()
        self.fit_window.setRange(5, 1000)
        self.fit_window.setValue(30)
        self.fit_window.setSuffix(" px")
        self.fit_ellipse = QtWidgets.QCheckBox("Show 1/e² ellipse and axes")
        self.fit_ellipse.setChecked(True)
        for w in (self.fit_units, self.fit_origin):
            w.currentIndexChanged.connect(self._show_fit)
        self.fit_ellipse.toggled.connect(self._show_fit)
        fit_form = QtWidgets.QFormLayout()
        fit_form.addRow("Units", self.fit_units)
        fit_form.addRow("(0, 0) at", self.fit_origin)
        fit_form.addRow("Fit window ±", self.fit_window)
        fit_widget = QtWidgets.QWidget()
        fit_widget.setLayout(fit_form)

        # dead pixels: found on demand from the lit sensor, kept in a JSON file, ignored in fits
        self.dead_label = QtWidgets.QLabel()
        self.dead_label.setStyleSheet("color: gray")
        self.dead_label.setWordWrap(True)
        find_dead = QtWidgets.QPushButton("Find dead pixels…")
        find_dead.setToolTip("Light the whole sensor with a broad beam and read out the full sensor first.\n"
                             "Averages 8 frames, maps dead pixels and dark spots (dust), saves the map.")
        find_dead.clicked.connect(lambda: self.find_dead_pixels())
        self.dead_ignore = QtWidgets.QCheckBox("Ignore them in fits")
        self.dead_show = QtWidgets.QCheckBox("Show them on the image")
        self.dead_show.setToolTip("Circles: dead pixels; dots: dark-spot pixels")
        self.dead_show.toggled.connect(self._show_dead)

        self.log_scale = QtWidgets.QCheckBox("Logarithmic colour scale")
        self.log_scale.setChecked(True)
        self.log_scale.toggled.connect(self._set_colour_scale)

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(self._group("Exposure", self.exposure))
        layout.addWidget(self._group("Gain", self.gain))
        layout.addWidget(self._group("Readout region (camera)", roi_widget, apply, full, self.show_roi))
        layout.addWidget(self._group("Background", self.bg_label, record, load, self.subtract, self.bg_warning))
        layout.addWidget(self._group("Gaussian fit", self.fit_on, fit_widget, self.fit_ellipse,
                                     self.dead_label, find_dead, self.dead_ignore, self.dead_show))
        layout.addWidget(self._group("Display", self.log_scale))
        layout.addStretch(1)
        # The settings live in a dock: scrollable (never taller than the screen), draggable to the
        # other side, and floatable into a window of its own (drag its title bar out, or double-click it).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(self.PANEL_MIN_WIDTH)
        self.dock = QtWidgets.QDockWidget("Settings", self)
        self.dock.setObjectName("settings")           # needed to save / restore the layout
        self.dock.setWidget(scroll)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                              | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                              | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
                                  | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.resizeDocks([self.dock], [self.PANEL_MIN_WIDTH], QtCore.Qt.Orientation.Horizontal)

        # toolbar: show / hide the settings, put the layout back
        self.toolbar.addSeparator()
        self.act_settings = self.dock.toggleViewAction()
        self.act_settings.setText("Settings")
        self.act_settings.setToolTip("Show / hide the settings column (drag its title bar out for a separate window)")
        self.toolbar.addAction(self.act_settings)
        act_reset = QtGui.QAction("Reset layout", self)
        act_reset.setToolTip("Settings docked on the left, window maximised")
        act_reset.triggered.connect(self.reset_layout)
        self.toolbar.addAction(act_reset)

    def _build_image(self):
        W, H = self.cam.sensor_width, self.cam.sensor_height
        self.view = pg.GraphicsLayoutWidget()
        centre = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(centre)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(self.view, 1)
        # Rows under the plot: pixel under the mouse and its value; the Gaussian fit.
        # Plain labels that wrap, so a long readout never makes the window wider than the screen.
        self.readout, self.fit_readout = QtWidgets.QLabel(""), QtWidgets.QLabel("")
        for label, size in ((self.readout, 10), (self.fit_readout, 12)):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            label.setContentsMargins(8, 0, 8, 2)
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Minimum)
            font = label.font()
            font.setPointSize(size)
            label.setFont(font)
            column.addWidget(label)
        self.setCentralWidget(centre)
        self.plot = self.view.addPlot(viewBox=_PanViewBox())
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)                       # row 0 at the top, like the sensor
        self.plot.setLabel("bottom", "x (px)")
        self.plot.setLabel("left", "y (px)")
        self.plot.setLimits(xMin=-0.1 * W, xMax=1.1 * W, yMin=-0.1 * H, yMax=1.1 * H)

        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.bar = pg.ColorBarItem(values=(0, self.cam.max_value), colorMap=colour_map(self.cam.max_value),
                                   interactive=False, label="counts")
        self.bar.setImageItem(self.img_item, insert_in=self.plot)

        self.roi = pg.RectROI([0, 0], [W, H], pen=pg.mkPen("y", width=2), sideScalers=True,
                              maxBounds=QtCore.QRectF(0, 0, W, H))
        self.roi.setZValue(10)
        self.plot.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self._roi_moved)
        self._set_roi(**self._aoi)
        self.plot.autoRange()

        self.fit_curve = pg.PlotDataItem(pen=pg.mkPen("w", width=2), connect="finite")
        self.fit_curve.setZValue(11)
        self.plot.addItem(self.fit_curve)
        self._fit = None          # last fit, in sensor pixel coordinates
        self._fit_time = 0.0
        self.dead_marks = pg.ScatterPlotItem(pen=pg.mkPen("r", width=1.5), brush=None, symbol="o", size=12)
        self.spot_marks = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush("r"), symbol="s", size=4)
        for item in (self.dead_marks, self.spot_marks):
            item.setZValue(12)
            self.plot.addItem(item)
        self._load_dead_pixels()
        self._mouse_pos = None
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

    # ---------- Camera control ----------

    def _camera_call(self, func, *args, **kwargs) -> bool:
        """Call into the camera; report a failure instead of crashing the viewer."""
        try:
            func(*args, **kwargs)
            return True
        except Exception as err:  # driver errors are RuntimeError / ValueError
            QtWidgets.QMessageBox.warning(self, "Camera", str(err))
            return False

    def set_live(self, on: bool):
        if on:
            self._camera_call(self.cam.start_live)
            self.timer.start()
        else:
            self.timer.stop()
            self._camera_call(self.cam.stop_live)
        self.act_live.setIcon(self._icon("SP_MediaPause" if on else "SP_MediaPlay"))

    def _set_exposure(self, exposure_us: float):
        self._camera_call(self.cam.set_exposure, exposure_us)
        self.exposure.setValue(self.cam.exposure_us)  # what the camera actually set
        self._check_background_settings()

    def _set_gain(self, gain_db: float):
        self._camera_call(self.cam.set_gain, gain_db)
        self.gain.setValue(self.cam.gain_db)
        self._check_background_settings()

    def _apply_roi(self):
        x, y, w, h = (spin.value() for spin in self.roi_spins)
        if self._camera_call(self.cam.set_aoi, offset_px=(x, y), width_px=w, height_px=h):
            self._aoi = self.cam.aoi
            self._set_roi(**self._aoi)            # the camera may have aligned it
            self._update_background_slice()
            self.plot.setRange(self._aoi_rect(), padding=0.05)

    def _full_frame(self):
        if self._camera_call(self.cam.reset_aoi):
            self._aoi = self.cam.aoi
            self._set_roi(**self._aoi)
            self._update_background_slice()
            self.plot.autoRange()

    # ---------- Background ----------

    def record_background(self, path: str | Path | None = None):
        """Take one snapshot as the background (asks where to save it unless path is given)."""
        if path is None:
            default = self.data_root / "background" / f"background_{datetime.now():%Y_%m_%d-%H_%M_%S}.npy"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save background as", str(default),
                                                            "NumPy array (*.npy)")
            if not path:
                return
        path = Path(path)
        stem = path.stem.removeprefix("data_")   # snapshot() names the files data_<stem>.npy etc.
        try:
            img, folder = self.cam.snapshot("background", comment="background frame",
                                            path=path.parent, filename=stem)
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Background failed", str(err))
            return
        self._set_background(img, self.cam.aoi, folder / f"data_{stem}.npy",
                             (self.cam.exposure_us, self.cam.gain_db))

    def load_background(self, path: str | Path | None = None):
        """Use a saved data_*.npy as the background (its metadata_*.json gives the readout region)."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load background",
                                                            str(self.data_root / "background"),
                                                            "NumPy array (*.npy)")
            if not path:
                return
        path = Path(path)
        img = np.load(path)
        meta = path.with_name(path.name.replace("data_", "metadata_", 1)).with_suffix(".json")
        meta = json.loads(meta.read_text()) if meta.exists() else {}
        aoi = meta.get("aoi")
        settings = (meta["exposure_us"], meta["gain_db"]) if "exposure_us" in meta else None
        if aoi is None:  # plain array: accept the full sensor or the current readout
            h, w = img.shape
            if (w, h) == (self.cam.sensor_width, self.cam.sensor_height):
                aoi = dict(offset_x=0, offset_y=0, width=w, height=h)
            elif (w, h) == (self._aoi["width"], self._aoi["height"]):
                aoi = dict(self._aoi)
            else:
                QtWidgets.QMessageBox.warning(self, "Background", f"{path.name} is {w} × {h} px: neither the "
                                              "full sensor nor the current readout region.")
                return
        self._set_background(img, aoi, path, settings)

    def _set_background(self, img: np.ndarray, aoi: dict, path: Path, settings: tuple | None):
        self._background, self._background_aoi, self._background_path = img.astype(np.int32), aoi, Path(path)
        self._background_settings = settings
        self.bg_label.setText(path.name)
        self.bg_label.setToolTip(str(path))
        self.subtract.setEnabled(True)
        self.subtract.setChecked(True)
        self._update_background_slice()
        self._check_background_settings()

    def _check_background_settings(self):
        """Warn while the exposure or gain differs from the background's: the subtraction is then misleading."""
        text = ""
        if self._background is not None:
            if self._background_settings is None:
                text = "Exposure and gain of this background are unknown (no metadata)."
            else:
                exposure, gain = self._background_settings
                if abs(self.cam.exposure_us - exposure) > 0.5 or abs(self.cam.gain_db - gain) > 0.05:
                    text = (f"Background taken at {exposure:.0f} µs, {gain:.1f} dB; camera now at "
                            f"{self.cam.exposure_us:.0f} µs, {self.cam.gain_db:.1f} dB. "
                            "Subtraction is misleading: record a new background.")
        self.bg_warning.setText(text)
        self.bg_warning.setVisible(bool(text))
        if text:
            self.statusBar().showMessage(text, 10000)

    def _update_background_slice(self):
        """The part of the background covering the current readout region, or None."""
        self._bg_slice = None
        if self._background is None:
            return
        a, b = self._aoi, self._background_aoi
        x0, y0 = a["offset_x"] - b["offset_x"], a["offset_y"] - b["offset_y"]
        if x0 >= 0 and y0 >= 0 and x0 + a["width"] <= b["width"] and y0 + a["height"] <= b["height"]:
            self._bg_slice = self._background[y0:y0 + a["height"], x0:x0 + a["width"]]
        else:
            self.statusBar().showMessage("Background does not cover the current readout region: "
                                         "not subtracted", 8000)

    # ---------- ROI rectangle <-> spin boxes ----------

    def _aoi_rect(self) -> QtCore.QRectF:
        a = self._aoi
        return QtCore.QRectF(a["offset_x"], a["offset_y"], a["width"], a["height"])

    def _set_roi(self, offset_x, offset_y, width, height):
        self.roi.setPos((offset_x, offset_y), finish=False)
        self.roi.setSize((width, height), finish=False)
        self._roi_moved()  # the ROI only signals when its state changes; sync the spin boxes anyway

    def _roi_moved(self):
        values = (*self.roi.pos(), *self.roi.size())
        for spin, value in zip(self.roi_spins, values):
            spin.blockSignals(True)
            spin.setValue(int(round(value)))
            spin.blockSignals(False)

    def _spins_changed(self):
        self._set_roi(*(spin.value() for spin in self.roi_spins))

    # ---------- Display ----------

    def _set_colour_scale(self, log: bool):
        self.bar.setColorMap(colour_map(self.cam.max_value, log))

    def _mouse_moved(self, scene_pos):
        self._mouse_pos = scene_pos if self.plot.sceneBoundingRect().contains(scene_pos) else None
        self._update_readout()

    def _update_readout(self):
        """Sensor pixel under the mouse and its value in the frame on screen."""
        text = ""
        if self._mouse_pos is not None and self._last is not None:
            p = self.plot.getViewBox().mapSceneToView(self._mouse_pos)
            x, y = int(math.floor(p.x())), int(math.floor(p.y()))   # pixel i spans [i, i + 1)
            col, row = x - self._aoi["offset_x"], y - self._aoi["offset_y"]
            if 0 <= row < self._last.shape[0] and 0 <= col < self._last.shape[1]:
                text = f"pixel x = {x}, y = {y}      value = {self._last[row, col]} / {self.cam.max_value} counts"
                if self.subtract.isChecked() and self._bg_slice is not None:
                    text += "   (background subtracted)"
        self.readout.setText(text)

    def _poll(self):
        frame = self.cam.grab_latest()
        if frame is None:
            return
        if self._dead_accum is not None:
            self._accumulate_dead(frame)
        if self.subtract.isChecked() and self._bg_slice is not None and self._bg_slice.shape == frame.shape:
            frame = np.clip(frame.astype(np.int32) - self._bg_slice, 0, None).astype(np.uint16)
        self.img_item.setImage(frame, autoLevels=False)
        if frame.shape != self._shape:  # new AOI: place the image in sensor coordinates
            self._shape = frame.shape
            self.img_item.setRect(self._aoi_rect())
        self._last = frame
        self._frames += 1
        self._update_readout()  # the value under a resting mouse follows the live frame
        self._update_fit(frame)
        now = time.perf_counter()
        if now - self._t_status >= self.STATUS_S:
            self._update_status(now)

    # ---------- Gaussian fit ----------

    def _update_fit(self, frame: np.ndarray):
        """Fit the displayed frame every 1 / FIT_PER_SECOND s while the fit is on."""
        now = time.perf_counter()
        if not self.fit_on.isChecked() or now - self._fit_time < 1.0 / self.FIT_PER_SECOND:
            return
        if self.dead_ignore.isChecked() and self._dead is not None:
            frame = repair(frame, self._dead, offset=(self._aoi["offset_y"], self._aoi["offset_x"]))
        try:
            fit = fit_gaussian_2d(frame, self.fit_window.value())
        except RuntimeError as err:
            self._fit = dict(error=str(err))
        else:
            fit["x0"] += self._aoi["offset_x"]   # array coordinates of the frame -> sensor pixels
            fit["y0"] += self._aoi["offset_y"]
            fit["ms"] = 1e3 * (time.perf_counter() - now)
            self._fit = fit
        self._fit_time = time.perf_counter()   # interval counted from the end of the fit
        self._show_fit()

    def _show_fit(self):
        """Readout row and ellipse for the last fit, in the chosen units and origin."""
        fit = self._fit if self.fit_on.isChecked() else None
        if fit is None or "error" in fit:
            self.fit_readout.setText("" if fit is None else f"Gaussian fit: {fit['error']}")
            self.fit_curve.setData([], [])
            return
        scale, unit = self.fit_units.currentData(), "px" if self.fit_units.currentIndex() == 0 else "µm"
        x, y = fit["x0"], fit["y0"]
        if self.fit_origin.currentIndex() == 1:       # (0, 0) at the centre of the sensor
            x -= (self.cam.sensor_width - 1) / 2
            y -= (self.cam.sensor_height - 1) / 2
        self.fit_readout.setText(
            f"<b>Gaussian centre</b>  x = <b>{x * scale:.2f}</b>, y = <b>{y * scale:.2f}</b> {unit}"
            f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;1/e² radii {fit['wx'] * scale:.2f} × {fit['wy'] * scale:.2f} {unit}, "
            f"axis at {np.degrees(fit['theta']):.0f}°&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;fitted peak {fit['peak']:.0f} counts"
            f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;{fit['ms']:.0f} ms"
            + (f" (binned {fit['binning']}×)" if fit["binning"] > 1 else ""))
        if not self.fit_ellipse.isChecked():
            self.fit_curve.setData([], [])
            return
        # ellipse and its principal axes (each as long as its 1/e² radius) in plot coordinates
        # (pixel i spans [i, i + 1), so its centre is i + 0.5)
        cx, cy = fit["x0"] + 0.5, fit["y0"] + 0.5
        t = np.linspace(0, 2 * np.pi, 73)
        ex, ey = fit["wx"] * np.cos(t), fit["wy"] * np.sin(t)
        c, s_ = np.cos(fit["theta"]), np.sin(fit["theta"])
        wx, wy = fit["wx"], fit["wy"]
        xs = np.concatenate([cx + ex * c - ey * s_, [np.nan, cx - wx * c, cx + wx * c, np.nan, cx + wy * s_, cx - wy * s_]])
        ys = np.concatenate([cy + ex * s_ + ey * c, [np.nan, cy - wx * s_, cy + wx * s_, np.nan, cy - wy * c, cy + wy * c]])
        self.fit_curve.setData(xs, ys)

    # ---------- dead pixels ----------

    def _load_dead_pixels(self):
        if self._dead_path.exists():
            found = load_dead_pixels(self._dead_path)
            self._set_dead_pixels(found, f"found {found['info'].get('found', '?')[:16]}")
        else:
            self._set_dead_pixels(None, "no dead-pixel map yet")

    def _set_dead_pixels(self, found: dict | None, when: str):
        self._dead = found
        have = found is not None and (len(found["ys"]) > 0 or len(found["spots"]) > 0)
        text = when if found is None else (
            f"{len(found['ys'])} dead pixels, {len(found['spots'])} dark spots "
            f"({sum(s['n_pixels'] for s in found['spots'])} px), {when}")
        self.dead_label.setText(f"{text}<br><span style='font-size: 8pt'>{self._dead_path.name}</span>")
        self.dead_label.setToolTip(str(self._dead_path))
        for w in (self.dead_ignore, self.dead_show):
            w.setEnabled(have)
        self.dead_ignore.setChecked(have)
        self._show_dead()

    def _show_dead(self):
        if self.dead_show.isChecked() and self._dead is not None:
            ys, xs = self._dead["ys"], self._dead["xs"]
            self.dead_marks.setData(xs + 0.5, ys + 0.5)   # pixel i spans [i, i + 1)
            sy, sx = all_bad_pixels(dict(ys=[], xs=[], spots=self._dead["spots"]))
            self.spot_marks.setData(sx + 0.5, sy + 0.5)
        else:
            self.dead_marks.setData([], [])
            self.spot_marks.setData([], [])

    def find_dead_pixels(self):
        """Average DEAD_PIXEL_FRAMES raw frames of the full sensor, flag the dead pixels, save the map."""
        a = self._aoi
        if a["width"] != self.cam.sensor_width or a["height"] != self.cam.sensor_height:
            QtWidgets.QMessageBox.information(self, "Dead pixels", "Read out the full sensor first ('Full sensor' button): "
                                              "the map is for the whole chip.")
            return
        if not self.timer.isActive():
            QtWidgets.QMessageBox.information(self, "Dead pixels", "Start the live stream first.")
            return
        self._dead_accum = []
        self.statusBar().showMessage(f"Averaging {self.DEAD_PIXEL_FRAMES} frames for the dead-pixel map…")

    def _accumulate_dead(self, frame: np.ndarray):
        self._dead_accum.append(frame.astype(np.float32))
        if len(self._dead_accum) < self.DEAD_PIXEL_FRAMES:
            return
        stack = np.stack(self._dead_accum)
        self._dead_accum = None
        found = find_dead_pixels(stack)
        mean = stack.mean(axis=0)
        lit = float(np.percentile(mean, 90))
        if lit < 20 * found["noise_sigma"] + 5:
            QtWidgets.QMessageBox.warning(self, "Dead pixels", "The sensor is barely lit: a dead pixel cannot be told from "
                                          "a dark one. Light the whole chip with a broad beam and try again.")
            return
        save_dead_pixels(self._dead_path, found,
                         camera=getattr(self.cam, "sensor_name", type(self.cam).__name__),
                         serial=getattr(self.cam, "serial", None),
                         sensor=[self.cam.sensor_height, self.cam.sensor_width],
                         n_frames=self.DEAD_PIXEL_FRAMES, exposure_us=self.cam.exposure_us, gain_db=self.cam.gain_db,
                         level_90th_percentile=round(lit, 1))
        self._set_dead_pixels(found, "found just now")
        self.dead_show.setChecked(True)
        self.statusBar().showMessage(
            f"{len(found['ys'])} dead pixels, {len(found['spots'])} dark spots (noise σ {found['noise_sigma']:.2f} counts) "
            f"saved to {self._dead_path}", 15000)

    def _update_status(self, now: float):
        fps = self._frames / (now - self._t_status)
        self._frames, self._t_status = 0, now
        peak = int(self._last.max())
        saturated = peak >= 0.99 * self.cam.max_value
        a = self._aoi
        text = (f"{fps:.1f} fps shown (camera {self.cam.frame_rate():.1f})   |   "
                f"peak {peak} / {self.cam.max_value}   |   mean {self._last.mean():.1f}   |   "
                f"readout {a['width']} × {a['height']} px at ({a['offset_x']}, {a['offset_y']})")
        self.status.setText(("SATURATED   |   " if saturated else "") + text)
        self.status.setStyleSheet("color: red; font-weight: bold" if saturated else "")

    # ---------- Snapshot ----------

    def save_snapshot(self):
        dialog = SnapshotDialog(self, self.experiment, self.data_root)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.take_snapshot(*dialog.values())

    def take_snapshot(self, experiment: str, data_root: Path, comment: str = ""):
        """Save one snapshot through the camera's snapshot() (pauses the stream)."""
        self.experiment, self.data_root = experiment, Path(data_root)
        self.cam.data_root = self.data_root
        try:
            img, out_dir = self.cam.snapshot(experiment, comment=comment, save_preview=True)
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Snapshot failed", str(err))
            return
        # The data file is always the raw frame; note which background was in use
        for meta_path in Path(out_dir).glob("metadata_*.json"):
            meta = json.loads(meta_path.read_text())
            meta["background"] = None if self._background is None else dict(
                file=str(self._background_path), applied_to_data=False,
                shown_subtracted=self.subtract.isChecked())
            meta_path.write_text(json.dumps(meta, indent=4))
        peak = int(img.max())
        warning = "   —   SATURATED, lower the exposure" if peak >= 0.99 * self.cam.max_value else ""
        self.statusBar().showMessage(f"Saved {out_dir}   (peak {peak} / {self.cam.max_value}){warning}", 15000)

    # ---------- window layout ----------

    _settings = ("cameras-beam-profilers", "live_view")

    def show_remembered(self):
        """Show the window as it was closed last time (size, position, settings dock); maximised the first time."""
        store = QtCore.QSettings(*self._settings)
        geometry, state = store.value("geometry"), store.value("state")
        if geometry is not None and self.restoreGeometry(geometry):
            if state is not None:
                self.restoreState(state)
            self.show()
        else:
            self.showMaximized()   # fills the screen minus the taskbar, so nothing is hidden behind it

    def reset_layout(self):
        self.dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.dock.show()
        self.resizeDocks([self.dock], [self.PANEL_MIN_WIDTH], QtCore.Qt.Orientation.Horizontal)
        self.showNormal()
        self.showMaximized()

    def closeEvent(self, event):
        store = QtCore.QSettings(*self._settings)
        store.setValue("geometry", self.saveGeometry())
        store.setValue("state", self.saveState())
        self.set_live(False)
        super().closeEvent(event)


def run_live_view(cam, data_root: str | Path, experiment: str = "live view") -> int:
    """Open the viewer on `cam` and block until the window is closed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LiveView(cam, data_root, experiment)
    window.show_remembered()
    try:
        return app.exec()
    finally:
        window.set_live(False)
