"""
Thorlabs Zelux camera (CS165MU / CS165MU1: 1440 x 1080 Sony IMX273 global-
shutter mono sensor, 3.45 um pixels, 10-bit, USB 3.0) through the Thorlabs
Scientific Camera SDK.

The SDK's native DLL comes with ThorImageCAM / ThorCam; this class talks to it
through ctypes (_tsi_api.py), so Thorlabs' Python package is not needed.

Same snapshot interface as cinogy_uc480.CinCamUC480 and allied_vision.AlviumG1:
- camera configuration is slow and happens once
- snapshots are fast and never query hardware
- all camera metadata is cached internally

Plus the live mode of camera_live_view: start_live() / grab_latest() /
stop_live(), set_aoi() / reset_aoi(), exposure_range_us() / gain_range_db(),
and reconnect() for a camera whose USB link dropped.
"""

import ctypes as C
import json
import time

import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from . import _tsi_api as api
except ImportError:              # the file is run directly (python zelux.py, or the IDE's run button)
    import _tsi_api as api


class ThorlabsZelux:
    """
    Snapshot-oriented camera class.
    """

    # ---------- Declarative parameter registry ----------
    _PARAMETERS = {
        "black_level": {
            "default": None,  # None = keep the camera's default
            "setter": "set_black_level",
        },
        "exposure_us": {
            "default": 6000,
            "setter": "set_exposure",
        },
        "gain_db": {
            "default": 0.0,
            "setter": "set_gain",
        },
    }

    _DISCOVERY_S = 3.0       # a camera just plugged in (or back from a USB reset) takes a moment to show up
    # Frame buffers the SDK fills (arm). Thorlabs' examples use 2, which is fine on a USB 3 link but
    # gets no frame through at all on USB 2 (behind a hub, measured on a CS165MU1): with 8, full
    # frames arrive (~15 fps, half of them lost in transfer, the newest is shown anyway).
    _BUFFERS = 8
    # Snapshot: re-trigger when no frame arrives within exposure + _RETRIGGER_MS, for up to
    # exposure + _SNAPSHOT_TIMEOUT_MS. On USB 2 the first few triggers after opening get nothing.
    _RETRIGGER_MS = 1000
    _SNAPSHOT_TIMEOUT_MS = 10000
    _LOST_WARNING = 0.1      # warn (once per stream) when more than this fraction of live frames is lost

    # ---------- Initialization ----------

    def __init__(self, config: dict):
        """
        Initialize camera and apply configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

            Required keys
            -------------
            data_root : str
                Root directory where snapshot data will be stored.

            Optional keys
            -------------
            camera_id : str or None
                Serial number of the camera to open; None opens the first one found.
            dll_path : str or None
                Path to thorlabs_tsi_camera_sdk.dll. None searches the
                THORLABS_TSI_SDK_DLL environment variable and the ThorImageCAM /
                ThorCam install folders.
            exposure_us : float or None
            gain_db : float or None
            black_level : int or None
                Black-level offset (counts); None keeps the camera's.
        """
        self._cfg = config

        self.data_root = Path(self._cfg["data_root"])
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.camera_id = self._cfg.get("camera_id", None)
        self._h = None
        self._live = False
        self._fps, self._fps_mark = 0.0, None
        self._lost_warned = False
        self._camera_metadata = {}

        self._dll, self.dll_path = api.load_api(self._cfg.get("dll_path", None))
        api.open_sdk()
        self._sdk_open = True

        try:
            self._connect_camera()
            self._apply_initial_settings()
            # ---- Metadata cache (must come LAST) ----
            self._camera_metadata = self._query_static_metadata()
        except Exception:
            # Release the camera before re-raising, otherwise nothing else can open it
            try:
                self.close()
            except Exception:
                pass
            raise

    # ---------- Camera connection ----------

    def _call(self, name: str, *args, what: str = None):
        api.check(getattr(self._dll, name)(self._h, *args), what or name)

    def _get(self, name: str, ctype=C.c_int):
        value = ctype()
        self._call(name, C.byref(value))
        return value.value

    def _get_string(self, name: str) -> str:
        buf = C.create_string_buffer(api.STRING_MAX)
        self._call(name, buf, api.STRING_MAX)
        return buf.value.decode(errors="replace").strip()

    def _wait_for_cameras(self) -> list[str]:
        """Serial numbers the SDK sees (the wanted one if camera_id is set), waiting up to _DISCOVERY_S."""
        deadline = time.perf_counter() + self._DISCOVERY_S
        while True:
            serials = api.discover()
            if self.camera_id is not None:
                serials = [s for s in serials if s == str(self.camera_id)]
            if serials or time.perf_counter() > deadline:
                return serials
            time.sleep(0.25)

    def _connect_camera(self):
        """
        Open the camera (the one with serial camera_id, or the first found).
        """
        serials = self._wait_for_cameras()
        if not serials:
            wanted = f"Thorlabs camera with serial {self.camera_id}" if self.camera_id else "Thorlabs camera"
            raise RuntimeError(
                f"[Zelux] No {wanted} found. Is it plugged in (Device Manager: 'Thorlabs Camera Zelux')?\n"
                f"  Use a USB 3 port directly on the PC if it keeps dropping off."
            )
        h = C.c_void_p()
        ret = self._dll.tl_camera_open_camera(serials[0].encode(), C.byref(h))
        if ret != 0:
            raise RuntimeError(
                f"[Zelux] Could not open camera {serials[0]} (error {ret}): {api.last_error()}\n"
                f"  ThorImageCAM, ThorCam or another script likely has it open.\n"
                f"  Close that application and run again."
            )
        self._h = h

        self.serial = self._get_string("tl_camera_get_serial_number")
        self.camera_id = self.serial  # reconnect() reopens this same camera
        self.model = self._get_string("tl_camera_get_model")
        self.sensor_name = self.model
        self.firmware = self._get_string("tl_camera_get_firmware_version")
        self.sensor_type = api.SENSOR_TYPES.get(self._get("tl_camera_get_camera_sensor_type"), "unknown")
        self.sensor_width = self._get("tl_camera_get_sensor_width")
        self.sensor_height = self._get("tl_camera_get_sensor_height")
        self.pixel_size_um = round(self._get("tl_camera_get_sensor_pixel_width", C.c_double), 4)
        self.bit_depth = self._get("tl_camera_get_bit_depth")
        self.max_value = 2 ** self.bit_depth - 1

        # Frames start on a software trigger; sensor pixels, no binning
        self._call("tl_camera_set_operation_mode", api.OPERATION_MODE_SOFTWARE_TRIGGERED)
        self._call("tl_camera_set_binx", 1)
        self._call("tl_camera_set_biny", 1)

        print(
            f"[Zelux] Connected: {self.model} (serial {self.serial}), {self.sensor_width} x {self.sensor_height}, "
            f"{self.pixel_size_um:g} um, {self.bit_depth}-bit, via {Path(self.dll_path).parent}"
        )
        # (no USB 2 check: the SDK's tl_camera_get_usb_port_type says USB 3.0 on a USB 2 hub port too)
        if self.sensor_type != "monochrome":
            print(f"[Zelux] WARNING: {self.sensor_type} sensor: frames are the raw sensor values, not demosaiced.")

    def close(self):
        try:
            if self._h is not None:
                self.stop_live()
                self._dll.tl_camera_close_camera(self._h)
                self._h = None
        finally:
            if self._sdk_open:
                api.close_sdk()
                self._sdk_open = False

    def reconnect(self):
        """
        Reopen the camera after its USB link was lost (cable unplugged and back,
        USB reset after standby). Settings go back to the config's; the live
        viewer then restores its own exposure, gain and readout region.
        """
        self._live = False
        if self._h is not None:
            self._dll.tl_camera_disarm(self._h)
            self._dll.tl_camera_close_camera(self._h)
            self._h = None
        self._connect_camera()
        self._apply_initial_settings()
        print("[Zelux] Reconnected")

    # ---------- Area of interest ----------

    @property
    def aoi(self) -> dict:
        """Readout rectangle in sensor pixels: offset_x, offset_y, width, height."""
        return dict(self._camera_metadata["aoi"])

    def _apply_roi(self, x0: int, y0: int, x1: int, y1: int):
        """Set the readout rectangle (corners inclusive), pausing a running live stream for it."""
        was_live = self.live
        self.stop_live()
        self._call("tl_camera_set_roi", x0, y0, x1, y1)
        ulx, uly, lrx, lry = (C.c_int() for _ in range(4))
        self._call("tl_camera_get_roi", C.byref(ulx), C.byref(uly), C.byref(lrx), C.byref(lry))
        self._width = self._get("tl_camera_get_image_width")
        self._height = self._get("tl_camera_get_image_height")
        self._camera_metadata["aoi"] = dict(offset_x=ulx.value, offset_y=uly.value,
                                            width=self._width, height=self._height)
        if was_live:
            self.start_live()

    def reset_aoi(self):
        """Read out the full sensor."""
        self._apply_roi(0, 0, self.sensor_width - 1, self.sensor_height - 1)

    def set_aoi(
            self,
            offset_px: tuple[int, int] = (0, 0),
            width_px: int = None,
            height_px: int = None,
    ):
        """
        Restrict the sensor readout to a rectangle. offset_px is the top-left
        corner (column, row), with the origin at the sensor's top-left and y
        down. The rectangle is clamped to what the sensor allows: read `aoi`
        for what was set.
        """
        if width_px is None or height_px is None:
            raise ValueError("set_aoi() needs width_px and height_px.")
        lim = [C.c_int() for _ in range(8)]
        self._call("tl_camera_get_roi_range", *(C.byref(v) for v in lim))
        ulx_min, uly_min, lrx_min, lry_min, ulx_max, uly_max, lrx_max, lry_max = (v.value for v in lim)

        x0 = int(np.clip(offset_px[0], ulx_min, ulx_max))
        y0 = int(np.clip(offset_px[1], uly_min, uly_max))
        x1 = int(np.clip(x0 + int(width_px) - 1, lrx_min, lrx_max))
        y1 = int(np.clip(y0 + int(height_px) - 1, lry_min, lry_max))
        self._apply_roi(x0, y0, x1, y1)
        a = self.aoi
        print(f"[Zelux] AOI set: offset=({a['offset_x']}, {a['offset_y']}), size=({a['width']}x{a['height']})")

    # ---------- Declarative configuration ----------

    def _apply_initial_settings(self):
        """
        Apply all camera parameters from the configuration dictionary.
        """
        for param, spec in self._PARAMETERS.items():
            value = self._cfg.get(param, None)
            if value is None:
                value = spec.get("default", None)
            if value is None:
                continue
            getattr(self, spec["setter"])(value)

        self.reset_aoi()

    # ---------- Metadata cache ----------

    @property
    def dead_pixel_file(self) -> Path:
        """JSON map of this camera's dead pixels (written by the live viewer), next to the driver."""
        return Path(__file__).resolve().parent / f"dead_pixels_{self.serial}.json"

    def _query_static_metadata(self) -> dict:
        """
        Query camera metadata once and return JSON-safe values.
        """
        return {
            "camera_model": f"Thorlabs {self.model}",
            "driver": self.dll_path,
            "sensor_type": self.sensor_type,
            "serial": self.serial,
            "camera_id": self.camera_id,
            "firmware": self.firmware,
            "pixel_format": f"Mono{self.bit_depth}",
            "bit_depth": self.bit_depth,
            "pixel_size_um": self.pixel_size_um,

            "resolution": [self.sensor_height, self.sensor_width],
            "aoi": self.aoi,
            "dead_pixel_file": str(self.dead_pixel_file) if self.dead_pixel_file.exists() else None,

            "exposure_us": self._get_exposure_us(),
            "gain_db": self._get_gain_db(),
            "black_level": self._get("tl_camera_get_black_level"),
        }

    def _get_exposure_us(self) -> float:
        return float(self._get("tl_camera_get_exposure_time", C.c_longlong))

    def _gain_index_range(self) -> tuple[int, int]:
        lo, hi = C.c_int(), C.c_int()
        self._call("tl_camera_get_gain_range", C.byref(lo), C.byref(hi))
        return lo.value, hi.value

    def _index_to_db(self, index: int) -> float:
        db = C.c_double()
        self._call("tl_camera_convert_gain_to_decibels", int(index), C.byref(db))
        return round(db.value, 3)

    def _get_gain_db(self) -> float:
        lo, hi = self._gain_index_range()
        return 0.0 if hi <= lo else self._index_to_db(self._get("tl_camera_get_gain"))

    # ---------- Settable ranges and current values (for viewers) ----------

    @property
    def exposure_us(self) -> float:
        """Exposure actually in force (us), from the cache."""
        return self._camera_metadata["exposure_us"]

    @property
    def gain_db(self) -> float:
        return self._camera_metadata["gain_db"]

    def exposure_range_us(self) -> tuple[float, float]:
        lo, hi = C.c_longlong(), C.c_longlong()
        self._call("tl_camera_get_exposure_time_range", C.byref(lo), C.byref(hi))
        return float(lo.value), float(hi.value)

    def gain_range_db(self) -> tuple[float, float]:
        lo, hi = self._gain_index_range()
        return (0.0, 0.0) if hi <= lo else (self._index_to_db(lo), self._index_to_db(hi))

    # ---------- Parameter wrappers (update cache) ----------

    def set_exposure(self, exposure_us: float):
        """
        Set camera exposure time (microseconds). The camera takes whole microseconds.
        """
        lo, hi = self.exposure_range_us()
        self._call("tl_camera_set_exposure_time", int(round(np.clip(exposure_us, lo, hi))))

        actual_us = self._get_exposure_us()
        if abs(actual_us - exposure_us) > max(0.01 * exposure_us, 50):
            print(f"[Zelux] WARNING: requested {exposure_us} us exposure, camera set {actual_us:.0f} us")
        self._camera_metadata["exposure_us"] = actual_us

    def set_gain(self, gain_db: float):
        """
        Set camera gain (dB). The SDK sets a gain index; the dB value is converted both ways.
        """
        lo, hi = self._gain_index_range()
        if hi <= lo:
            print("[Zelux] Camera has no adjustable gain, ignoring gain setting.")
            self._camera_metadata["gain_db"] = 0.0
            return
        db_lo, db_hi = self._index_to_db(lo), self._index_to_db(hi)
        index = C.c_int()
        self._call("tl_camera_convert_decibels_to_gain", float(np.clip(gain_db, db_lo, db_hi)), C.byref(index))
        self._call("tl_camera_set_gain", int(np.clip(index.value, lo, hi)))

        actual_db = self._get_gain_db()
        if abs(actual_db - gain_db) > 0.1:
            print(f"[Zelux] WARNING: requested {gain_db} dB gain, camera set {actual_db:.2f} dB")
        self._camera_metadata["gain_db"] = actual_db

    def set_black_level(self, level: int):
        """
        Set the black-level offset (counts), so the dark level stays the same between snapshots.
        """
        lo, hi = C.c_int(), C.c_int()
        self._call("tl_camera_get_black_level_range", C.byref(lo), C.byref(hi))
        value = int(np.clip(level, lo.value, hi.value))
        if value != level:
            print(f"[Zelux] WARNING: black level {level} out of range [{lo.value}, {hi.value}], set {value}")
        self._call("tl_camera_set_black_level", value)
        self._camera_metadata["black_level"] = value

    # ---------- Image acquisition ----------

    def _pending_frame(self) -> np.ndarray | None:
        """Copy of the next frame the SDK holds, or None (waits up to the image poll timeout)."""
        buf = C.POINTER(C.c_ushort)()
        count, meta, meta_size = C.c_int(), C.POINTER(C.c_ubyte)(), C.c_int()
        self._call("tl_camera_get_pending_frame_or_null",
                   C.byref(buf), C.byref(count), C.byref(meta), C.byref(meta_size))
        if not buf:
            return None
        if self._live:
            self._count_frame(count.value)
        # the SDK reuses its buffer: copy now
        return np.ctypeslib.as_array(buf, shape=(self._height, self._width)).copy()

    def capture_frame(self) -> np.ndarray:
        """
        Acquire a single image frame (software-triggered single capture).

        A running live stream is paused for it, so the frame is exposed with
        the settings in force at the call.

        Returns
        -------
        np.ndarray
            Raw camera image (uint16, values 0 to self.max_value).
        """
        if self.live:
            self.stop_live()
            try:
                return self.capture_frame()
            finally:
                self.start_live()

        # Every trigger comes after this call, so whichever frame arrives has the current settings
        exposure_ms = self.exposure_us / 1000
        t_start = time.perf_counter()
        deadline = t_start + (exposure_ms + self._SNAPSHOT_TIMEOUT_MS) / 1000
        self._call("tl_camera_set_frames_per_trigger_zero_for_unlimited", 1)
        self._call("tl_camera_set_image_poll_timeout", int(exposure_ms + self._RETRIGGER_MS))
        self._call("tl_camera_arm", self._BUFFERS)
        frame, triggers = None, 0
        try:
            while frame is None and time.perf_counter() < deadline:
                self._call("tl_camera_issue_software_trigger")
                triggers += 1
                frame = self._pending_frame()
        finally:
            self._dll.tl_camera_disarm(self._h)
        if frame is None:
            raise RuntimeError(
                f"[Zelux] No frame reached the PC after {triggers} triggers in {time.perf_counter() - t_start:.0f} s: "
                f"the USB link loses them. Plug the camera into a USB 3 port directly (not a USB 2 port or hub)."
            )
        return frame

    # ---------- Live acquisition ----------
    # Free-running mode for viewers: one software trigger starts unlimited
    # frames; the SDK queues up to _BUFFERS of them and grab_latest() hands out
    # the newest without blocking (image poll timeout 0).

    @property
    def live(self) -> bool:
        return self._live

    def start_live(self):
        """Start free-running capture (no-op if running)."""
        if self._live:
            return
        self._call("tl_camera_set_frames_per_trigger_zero_for_unlimited", 0)
        self._call("tl_camera_set_image_poll_timeout", 0)
        self._call("tl_camera_arm", self._BUFFERS)
        self._call("tl_camera_issue_software_trigger")
        self._live = True
        self._fps, self._fps_mark = 0.0, None

    def stop_live(self):
        """Stop the stream (no-op if not running)."""
        if not self._live:
            return
        self._live = False
        self._dll.tl_camera_disarm(self._h)

    def grab_latest(self) -> np.ndarray | None:
        """
        The newest live frame, or None if none finished since the last call.
        Never blocks; frames that queued up in between are skipped.
        """
        newest = None
        while (frame := self._pending_frame()) is not None:
            newest = frame
        return newest

    def _count_frame(self, number: int):
        """Frames reaching the PC per second, over ~1 s (the SDK's measured frame rate reads 0 on
        the Zelux). The camera numbers its frames, so a gap in the numbers is a frame lost in transfer."""
        now = time.perf_counter()
        if self._fps_mark is None:
            self._fps_mark, self._delivered = (now, number), 0
            return
        self._delivered += 1
        t0, n0 = self._fps_mark
        if now - t0 >= 1.0:
            self._fps = self._delivered / (now - t0)
            made, lost = number - n0, number - n0 - self._delivered
            if not self._lost_warned and lost > self._LOST_WARNING * made:
                print(f"[Zelux] WARNING: {lost} of {made} frames lost on the way to the PC in {now - t0:.1f} s: "
                      f"the USB link is too slow (USB 2 port or hub?). Plug the camera into a USB 3 port "
                      f"directly for the full frame rate.")
                self._lost_warned = True
            self._fps_mark, self._delivered = (now, number), 0

    def frame_rate(self) -> float:
        """Frames per second reaching the PC (live mode)."""
        return self._fps

    # ---------- Snapshot saving ----------

    def snapshot(
        self,
        experiment_name: str,
        comment: str = "",
        save_preview: bool = False,
        save_metadata: bool = True,
        path: str | Path = None,
        filename: str = None,
        exposure_time_us: int = None,
        cropping_params: dict = None,
        plot_psf_3d: bool = False,
    ) -> tuple[np.ndarray, Path]:
        """
        Acquire and save a single snapshot with cached metadata.
        """
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        base_dir = self.data_root / experiment_name / f"snapshot_{timestamp}" if path is None else Path(path)
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = timestamp if filename is None else filename

        if exposure_time_us is not None:
            self.set_exposure(exposure_time_us)
            print("set exposure time to given value", exposure_time_us)

        if cropping_params is not None:
            self.set_aoi(**cropping_params)
        try:
            img = self.capture_frame()
            metadata = {**self._camera_metadata, "timestamp": timestamp, "comment": comment}
        finally:
            if cropping_params is not None:
                self.reset_aoi()

        if exposure_time_us is not None:
            default_us = self._cfg.get("exposure_us") or self._PARAMETERS["exposure_us"]["default"]
            self.set_exposure(default_us)
            print("set back exposure time us to default value", default_us)

        np.save(base_dir / f"data_{stem}.npy", img)

        if save_preview:
            # Shared PSF preview plots live in the allied_vision package for now
            from allied_vision.camera_plot_helpers import save_psf_2d, save_psf_3d
            title = f"{experiment_name}\n{timestamp}"
            save_psf_2d(img, base_dir / f"preview_{stem}.png", title=title, max_value=self.max_value)
            if plot_psf_3d:
                name_3d = f"preview_3d_{filename}_{timestamp}.png" if filename is not None else f"preview_3d_{timestamp}.png"
                save_psf_3d(img, base_dir / name_3d, title=title, max_value=self.max_value)

        if save_metadata:
            with open(base_dir / f"metadata_{stem}.json", "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"[Zelux] Snapshot saved to {base_dir}")
        return img, base_dir


if __name__ == "__main__":
    # Quick check that the camera answers: connect, grab one frame, report, close.
    # Real use goes through the examples (examples/live_view.py, examples/snapshot.py).
    cam = ThorlabsZelux({"data_root": Path.cwd() / "data"})
    try:
        img = cam.capture_frame()
        print(f"Frame {img.shape[1]} x {img.shape[0]}, peak {int(img.max())} / {cam.max_value}, "
              f"exposure {cam.exposure_us:.0f} us, gain {cam.gain_db:g} dB")
    finally:
        cam.close()
