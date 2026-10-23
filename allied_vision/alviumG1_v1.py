"""
Minimal camera abstraction for Allied Vision cameras using vmbpy.

Purpose
-------
This class provides a clean interface to:
- configure camera parameters declaratively
- acquire single frames (fast)
- save snapshot data with cached metadata

Design principles
-----------------
- Camera configuration is slow and happens once
- Snapshots are fast and never query hardware
- All camera metadata is cached internally

Live mode (for camera_live_view): start_live() streams frames into a
one-frame buffer that grab_latest() empties; stop_live() ends it. Exposure
and gain can be changed while streaming; set_aoi() / reset_aoi() restart the
stream around the change. The properties exposure_us, gain_db, aoi,
sensor_width / sensor_height, pixel_size_um (from the config, the camera does
not report it) and the ranges exposure_range_us() / gain_range_db() complete
the interface. Alvium and Mako (GigE) cameras share it; features whose names
differ between the families are looked up by _feature().
"""

import json
import threading
import time

import numpy as np
from pathlib import Path
from datetime import datetime
# import cv2 as cv   # kept for reference — no longer used for preview generation
import vmbpy as vpy
import matplotlib.pyplot as plt
from vmbpy import AccessMode


class AlviumG1:
    """
    Snapshot-oriented camera class.
    """

    # ---------- Pixel format ----------
    # Candidates for pixel_format="auto", highest bit depth first
    _MONO_FORMATS = ("Mono16", "Mono14", "Mono12", "Mono10", "Mono8")

    # ---------- Declarative parameter registry ----------
    _PARAMETERS = {
        "exposure_auto": {
            "default": False,
            "setter": "set_exposure_auto",
        },
        # "access_mode": {
        #     "default": "Full",
        #     "setter": "set_access_mode",
        # },
        "pixel_format": {
            "default": "auto",
            "setter": "set_pixel_format",
        },
        "gain_auto": {
            "default": False,
            "setter": "set_gain_auto",
        },
        "exposure_us": {
            "default": 6000,
            "setter": "set_exposure",
            "requires": lambda cfg: not cfg.get("exposure_auto", False),
        },
        "gain_db": {
            "default": 0.0,
            "setter": "set_gain",
            "requires": lambda cfg: not cfg.get("gain_auto", False),
        },
    }

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
            exposure_us : float or None
            gain_db : float or None
            exposure_auto : bool or None
            gain_auto : bool or None
            use_camera_roi : bool
                True (default) keeps the AOI the camera already has, e.g. one
                set in Vimba X Viewer. False resets to the full sensor.
            pixel_format : str or None
                "auto" (highest Mono bit depth the camera supports) or an
                explicit format such as "Mono8", "Mono10", "Mono12".
        """
        # Store config
        self._cfg = config

        # Data directory
        self.data_root = Path(self._cfg["data_root"])
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Camera ID
        self.camera_id = self._cfg.get("camera_id", None)
        self._camera_metadata = {}

        # ---- Camera connection & setup ----
        self._vmb_ctx = vpy.VmbSystem.get_instance()
        self._vmb_ctx.__enter__()  # enter ONCE

        self.vmb = self._vmb_ctx

        try:
            self._connect_camera()
            self._last_timestamp = None

            # Apply user configuration (sets camera state)
            self._camera_metadata = {}
            self._apply_initial_settings()

            # ---- Metadata cache (must come LAST) ----
            self._camera_metadata = self._query_static_metadata()
        except Exception:
            # Release Vimba before re-raising, otherwise its threads keep Python from exiting
            try:
                self.close()
            except Exception:
                pass
            raise

    # ---------- Camera connection ----------

    def _connect_camera(self):
        """
        Connect to the Allied Vision camera.
        """
        if self.camera_id is not None:
            self.cam = self.vmb.get_camera_by_id(self.camera_id)
        else:
            # Ignore the fake cameras from the Vimba X Camera Simulator
            cams = [
                c for c in self.vmb.get_all_cameras()
                if "Simulator" not in c.get_interface_id()
            ]
            if not cams:
                raise RuntimeError(
                    "No Allied Vision cameras found. Check the camera is plugged in "
                    "and not open in another program (e.g. Vimba X Viewer)."
                )
            #here I decide which amera gets connected might need to be changed eventually
            self.cam = cams[0]
            self.camera_id = self.cam.get_id()

        print(self.cam.get_access_mode(), "\n now setting to Full")
        try:
            self.set_access_mode("Full")
            self.cam._open()  # explicit, once
        except vpy.VmbCameraError as exc:
            raise RuntimeError(
                f"[Camera] Could not open '{self.camera_id}' in Full access mode.\n"
                f"  Another process (e.g. Vimba Viewer) likely has the camera open.\n"
                f"  Close that application and run again.\n"
                f"  Underlying error: {exc}"
            ) from exc

        # Keep whatever AOI the camera already has (e.g. set in Vimba X Viewer)
        # unless the caller asked for the full sensor.
        if self._cfg.get("use_camera_roi", True):
            offx, offy, w, h = self.get_roi()
            if (w, h) != (self.cam.Width.get_range()[1], self.cam.Height.get_range()[1]):
                print(f"[Camera] Using the camera's ROI: {w}x{h} at ({offx}, {offy}). "
                      f"Set use_camera_roi=False for the full sensor.")
            else:
                print(f"AOI: {w} {h} {offx} {offy}")
        else:
            self._uncrop_camera()

    def close(self):
        try:
            if hasattr(self, "cam"):
                try:
                    self.stop_live()
                except vpy.VmbError:
                    pass
                try:
                    self.cam.set_pixel_format(vpy.PixelFormat.Mono8)
                except vpy.VmbError:
                    pass
                self.cam._close()
        finally:
            if hasattr(self, "_vmb_ctx"):
                self._vmb_ctx._shutdown()  # <-- NOT __exit__()


    # ---------- Declarative configuration ----------

    def get_roi(self) -> tuple[int, int, int, int]:
        """
        Current area of interest as (offset_x, offset_y, width, height), in pixels.
        """
        return (
            int(self.cam.OffsetX.get()),
            int(self.cam.OffsetY.get()),
            int(self.cam.Width.get()),
            int(self.cam.Height.get()),
        )

    def set_roi(self, offset_x: int, offset_y: int, width: int, height: int):
        """
        Set the area of interest directly, in the order the Alvium requires
        (offsets to zero, then size, then offsets).
        """
        self.cam.OffsetX.set(self.cam.OffsetX.get_range()[0])
        self.cam.OffsetY.set(self.cam.OffsetY.get_range()[0])
        self.cam.Width.set(int(width))
        self.cam.Height.set(int(height))
        self.cam.OffsetX.set(int(offset_x))
        self.cam.OffsetY.set(int(offset_y))

    def _uncrop_camera(self):
        # Reset offsets first (many cameras require this order)
        offx_min, offx_max = self.cam.OffsetX.get_range()
        offy_min, offy_max = self.cam.OffsetY.get_range()

        self.cam.OffsetX.set(offx_min)
        self.cam.OffsetY.set(offy_min)

        # Then set width/height to maximum
        w_min, w_max = self.cam.Width.get_range()
        h_min, h_max = self.cam.Height.get_range()

        self.cam.Width.set(w_max)
        self.cam.Height.set(h_max)

        print(
            "AOI:",
            self.cam.Width.get(),
            self.cam.Height.get(),
            self.cam.OffsetX.get(),
            self.cam.OffsetY.get(),
        )
    
    def _crop_camera(
            self,
            offset_px: tuple[int, int] = (0, 0),
            width_px: int = None,
            height_px: int = None,
            ):
        if width_px is None or height_px is None:
            print("[Camera] No cropping params given, skipping crop.")
            return

        offx_min, _ = self.cam.OffsetX.get_range()
        offy_min, _ = self.cam.OffsetY.get_range()
        w_min, w_max = self.cam.Width.get_range()
        h_min, h_max = self.cam.Height.get_range()

        w_inc = self.cam.Width.get_increment()
        h_inc = self.cam.Height.get_increment()
        ox_inc = self.cam.OffsetX.get_increment() or 1
        oy_inc = self.cam.OffsetY.get_increment() or 1

        # Clamp offset and size to valid hardware ranges, then align to increment
        ox = (int(np.clip(offset_px[0], offx_min, w_max - w_min)) // ox_inc) * ox_inc
        oy = (int(np.clip(offset_px[1], offy_min, h_max - h_min)) // oy_inc) * oy_inc
        w  = max((int(np.clip(width_px,  w_min, w_max - ox)) // w_inc) * w_inc, w_min)
        h  = max((int(np.clip(height_px, h_min, h_max - oy)) // h_inc) * h_inc, h_min)

        # Alvium requires: reset offsets → set size → set offsets
        self.cam.OffsetX.set(offx_min)
        self.cam.OffsetY.set(offy_min)
        self.cam.Width.set(w)
        self.cam.Height.set(h)
        self.cam.OffsetX.set(ox)
        self.cam.OffsetY.set(oy)

        print(f"[Camera] AOI set: offset=({ox}, {oy}), size=({w}x{h})")


    def _apply_initial_settings(self):
        """
        Apply all camera parameters from the configuration dictionary.
        """
        for param, spec in self._PARAMETERS.items():
            value = self._cfg.get(param, None)

            if value is None:
                value = spec.get("default", None)

            requires = spec.get("requires", None)
            if requires is not None and not requires(self._cfg):
                continue

            if value is None:
                continue

            setter = getattr(self, spec["setter"])
            setter(value)

    # ---------- Metadata cache ----------

    def _query_static_metadata(self) -> dict:
        """
        Query camera metadata once and return JSON-safe values.
        """
        metadata = {
            "camera_id": self.camera_id,
            "pixel_format": str(self.px_format),
            "bit_depth": self.bit_depth,

            "resolution": [
                int(self.cam.SensorHeight.get()),
                int(self.cam.SensorWidth.get()),
            ],

            "exposure_us": float(self._feature(self._EXPOSURE_TIME).get()),
            "gain_db": self.gain_db,
            "exposure_auto": str(f.get()) if (f := self._optional("ExposureAuto")) is not None else None,
            "gain_auto": str(f.get()) if (f := self._optional("GainAuto")) is not None else None,
        }
        return metadata

    # ---------- Parameter wrappers (update cache) ----------

    # Features whose name differs between camera families: (Alvium, Mako)
    _EXPOSURE_TIME = ("ExposureTime", "ExposureTimeAbs")

    def _feature(self, names: tuple):
        """
        Return the first feature in `names` that this camera provides.
        """
        for name in names:
            try:
                return self.cam.get_feature_by_name(name)
            except vpy.VmbFeatureError:
                pass
        raise AttributeError(f"Camera '{self.camera_id}' has none of the features {names}")

    def _optional(self, name: str):
        """The feature `name`, or None when this camera (or the Vimba simulator) lacks it."""
        try:
            return self.cam.get_feature_by_name(name)
        except vpy.VmbFeatureError:
            return None

    def set_exposure(self, exposure_us: float):
        """
        Set camera exposure time (microseconds).
        """
        if (auto := self._optional("ExposureAuto")) is not None:
            auto.set("Off")
        self._feature(self._EXPOSURE_TIME).set(exposure_us)

        self._camera_metadata["exposure_us"] = exposure_us
        self._camera_metadata["exposure_auto"] = False

    def set_gain(self, gain_db: float):
        """
        Set camera analog gain (dB).
        """

        if (auto := self._optional("GainAuto")) is not None:
            auto.set("Off")
        if (gain := self._optional("Gain")) is None:
            print("[Camera] This camera has no Gain feature; gain not set.")
            return
        gain.set(gain_db)

        self._camera_metadata["gain_db"] = gain_db
        self._camera_metadata["gain_auto"] = False

    def set_exposure_auto(self, enabled: bool = True):
        """
        Enable or disable automatic exposure.
        """
        if (auto := self._optional("ExposureAuto")) is not None:
            auto.set("Continuous" if enabled else "Off")
        self._camera_metadata["exposure_auto"] = enabled

    def set_gain_auto(self, enabled: bool = True):
        """
        Enable or disable automatic gain.
        """
        if (auto := self._optional("GainAuto")) is not None:
            auto.set("Continuous" if enabled else "Off")
        self._camera_metadata["gain_auto"] = enabled

    def set_pixel_format(self, format="auto"):
        """
        Set camera pixel format.

        "auto" picks the highest Mono bit depth the camera supports.
        An explicit name (e.g. "Mono10") must be supported by the camera.
        """
        supported = {str(f) for f in self.cam.get_pixel_formats()}
        mono_supported = [f for f in self._MONO_FORMATS if f in supported]

        if format == "auto":
            if not mono_supported:
                raise RuntimeError(
                    f"Camera supports no Mono pixel format. Available: {sorted(supported)}"
                )
            format = mono_supported[0]
        elif format not in supported:
            raise ValueError(
                f"Pixel format '{format}' not supported by this camera. "
                f"Mono formats available: {mono_supported}"
            )

        self.px_format = getattr(vpy.PixelFormat, format)
        self.cam.set_pixel_format(self.px_format)

        self.bit_depth = int(format.removeprefix("Mono"))
        self.max_value = 2 ** self.bit_depth - 1
        self._camera_metadata["pixel_format"] = format
        self._camera_metadata["bit_depth"] = self.bit_depth
        print(f"[Camera] Pixel format: {format} ({self.bit_depth}-bit)")

    def set_access_mode(self, mode="Full"):
        """
        Set camera access mode (i.e. Full, 1/4,...)
        """
        if "Full" == mode:
            self.cam.set_access_mode(AccessMode.Full)
        else:
            print("Access mode not supported/setup in AlviumG1 wrapper")
            pass

    # ---------- Image acquisition ----------

    def capture_frame(self) -> np.ndarray:
        """
        Acquire a single image frame using continuous acquisition.

        Returns
        -------
        np.ndarray
            Raw camera image (uint16, values 0 to self.max_value).
        """

        if self._live:                      # get_frame() is not allowed while streaming: take the next live frame
            t0 = time.perf_counter()
            img = self.grab_latest()
            while img is None and time.perf_counter() - t0 < 2.0:
                time.sleep(0.005)
                img = self.grab_latest()
            if img is None:
                raise RuntimeError("[Camera] No frame from the live stream within 2 s")
            return img

        prev_mode = self.cam.AcquisitionMode.get()
        self.cam.AcquisitionMode.set("Continuous")

        frame = self.cam.get_frame(timeout_ms=2000)
        if frame.get_pixel_format() != self.px_format:
            frame = frame.convert_pixel_format(self.px_format)

        self.cam.AcquisitionMode.set(prev_mode)

        img = frame.as_numpy_ndarray()[:, :, 0]
        return img.astype(np.uint16, copy=False)

    # ---------- Live mode ----------

    _live = False
    _lock = threading.Lock()

    def start_live(self):
        """Stream frames into a one-frame buffer (see grab_latest)."""
        if self._live:
            return
        self._latest, self._frame_count, self._t_live = None, 0, time.perf_counter()
        self.cam.AcquisitionMode.set("Continuous")
        self.cam.start_streaming(self._on_frame, buffer_count=5)
        self._live = True

    def stop_live(self):
        if self._live:
            self._live = False
            self.cam.stop_streaming()

    @property
    def live(self) -> bool:
        return self._live

    def _on_frame(self, cam, stream, frame):
        """vmbpy frame handler (runs in a Vimba thread): keep the newest complete frame."""
        try:
            if frame.get_status() == vpy.FrameStatus.Complete:
                converted = frame if frame.get_pixel_format() == self.px_format \
                    else frame.convert_pixel_format(self.px_format)
                img = converted.as_numpy_ndarray()[:, :, 0].astype(np.uint16)   # copies out of the buffer
                with self._lock:
                    self._latest = img
                    self._frame_count += 1
        finally:
            cam.queue_frame(frame)

    def grab_latest(self) -> np.ndarray | None:
        """The newest frame since the previous call, or None."""
        with self._lock:
            img, self._latest = self._latest, None
        return img

    def frame_rate(self) -> float:
        """Frames received per second since the previous call (measured)."""
        now = time.perf_counter()
        with self._lock:
            n, self._frame_count = self._frame_count, 0
        dt, self._t_live = now - self._t_live, now
        return n / dt if dt > 0 else 0.0

    # ---------- Viewer interface: current values, ranges, readout region ----------

    @property
    def exposure_us(self) -> float:
        return float(self._feature(self._EXPOSURE_TIME).get())

    @property
    def gain_db(self) -> float:
        try:
            return float(self.cam.Gain.get())
        except (vpy.VmbFeatureError, AttributeError):
            return 0.0

    def exposure_range_us(self) -> tuple[float, float]:
        lo, hi = self._feature(self._EXPOSURE_TIME).get_range()
        return float(lo), float(hi)

    def gain_range_db(self) -> tuple[float, float]:
        try:
            lo, hi = self.cam.Gain.get_range()
            return float(lo), float(hi)
        except (vpy.VmbFeatureError, AttributeError):
            return 0.0, 0.0

    @property
    def sensor_width(self) -> int:
        return int(self.cam.SensorWidth.get())

    @property
    def sensor_height(self) -> int:
        return int(self.cam.SensorHeight.get())

    @property
    def pixel_size_um(self) -> float | None:
        """From the config ("pixel_size_um"); Vimba does not report it."""
        return self._cfg.get("pixel_size_um")

    @property
    def aoi(self) -> dict:
        offset_x, offset_y, width, height = self.get_roi()
        return dict(offset_x=offset_x, offset_y=offset_y, width=width, height=height)

    def set_aoi(self, offset_px: tuple[int, int], width_px: int, height_px: int):
        """Readout region (aligned to the camera's increments); the stream is restarted around it."""
        was_live = self._live
        self.stop_live()
        self._crop_camera(offset_px, width_px, height_px)
        self._camera_metadata["aoi"] = self.aoi
        if was_live:
            self.start_live()

    def reset_aoi(self):
        was_live = self._live
        self.stop_live()
        self._uncrop_camera()
        self._camera_metadata["aoi"] = self.aoi
        if was_live:
            self.start_live()

    # ---------- Snapshot saving ----------

    def snapshot(
        self,
        experiment_name: str,
        comment: str = "",
        save_preview: bool = False,
        save_metadata : bool = True,
        path : str|Path = None,
        filename : str = None,
        calibrated_preview: bool = True,
        exposure_time_us: int = None,
        cropping_params: dict = None,
        plot_psf_3d: bool = False,

    ) -> tuple[np.ndarray, Path]:
        """
        Acquire and save a single snapshot with cached metadata.
        """
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        if path is None:
            base_dir = (
                self.data_root
                / experiment_name
                / f"snapshot_{timestamp}"
            )
        else:
            base_dir = Path(path)
        base_dir.mkdir(parents=True, exist_ok=True)

        if exposure_time_us is not None:
            self.set_exposure(exposure_time_us)
            print("set exposure time to given value", exposure_time_us)

        # With no cropping_params the camera's current ROI is used as-is
        previous_roi = self.get_roi()
        if cropping_params is not None:
            self._crop_camera(**cropping_params)
        try:
            roi_used = self.get_roi()
            img = self.capture_frame()
        finally:
            if cropping_params is not None:
                # Put back whatever ROI was set before, rather than uncropping
                self.set_roi(*previous_roi)

        if exposure_time_us is not None:
            exposure_us = self._cfg.get("exposure_us")
            if exposure_us is not None:
                default_value_us = exposure_us
            else:
                default_value_us = self._PARAMETERS["exposure_us"]["default"]
            self.set_exposure(exposure_us=default_value_us)
            print("set back exposure time us to defautl value",default_value_us)

        final_path = base_dir / (
                                f"data_{timestamp}.npy"
                                if filename is None else
                                f"data_{filename}.npy"
                            )
        np.save(final_path, img)

        if save_preview:
            preview_stem = (
                f"preview_{timestamp}" if filename is None else f"preview_{filename}"
            )
            from .camera_plot_helpers import save_psf_2d, save_psf_3d
            save_psf_2d(
                img,
                base_dir / f"{preview_stem}.png",
                title=f"{experiment_name}\n{timestamp}",
                max_value=self.max_value,
            )
            if plot_psf_3d:
                save_psf_3d(
                    img,
                    base_dir / f"preview_3d_{filename}_{timestamp}.png" if filename is not None else base_dir / f"preview_3d_{timestamp}.png",
                    title=f"{experiment_name}\n{timestamp}",
                    max_value=self.max_value,
                )
            # --- Legacy B&W PNG (replaced by PSF colormap above) ---
            # preview = img.astype(np.float32)
            # if calibrated_preview:
            #     preview /= 4095.0   # Mono12 physical scaling
            # else:
            #     preview /= preview.max()   # contrast stretch
            # preview = (preview * 65535).astype(np.uint16)
            # cv.imwrite(str(base_dir / f"{preview_stem}.png"), preview)
        if save_metadata :
            offset_x, offset_y, roi_w, roi_h = roi_used
            metadata = {
                **self._camera_metadata,
                # AOI this frame was taken with: needed to map pixel positions
                # back to sensor coordinates when the camera is cropped.
                "roi": {
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "width": roi_w,
                    "height": roi_h,
                },
                "timestamp": timestamp,
                "comment": comment,
            }

            with open(base_dir / (
                                f"metadata_{timestamp}.json"
                                if filename is None else
                                f"metadata_{filename}.json"
                            )
                    , "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"[Camera] Snapshot saved to {base_dir}")

        return img, base_dir


    def plot(
            self,
            experiment_name: str | None = None,
            npy_path: str | Path | None = None,
            cmap=None,
    ):
        """
        Plot a snapshot image.

        If npy_path is given, plot that file.
        Otherwise, plot the most recent snapshot of the given experiment.
        If experiment_name is None, uses the last snapshot taken.

        Parameters
        ----------
        experiment_name : str or None
            Experiment directory name.
        npy_path : str | Path or None
            Explicit path to a .npy file.
        cmap : matplotlib colormap or None
            Colormap to use.  Defaults to PSF_CMAP (black → blue → green)
            with a fixed 0 to self.max_value linear scale.  Pass any
            matplotlib colormap to override.

        """
        from .camera_plot_helpers import PSF_CMAP
        if cmap is None:
            cmap = PSF_CMAP

        # -------- Resolve image path --------
        if npy_path is not None:
            npy_path = Path(npy_path)
            exp_dir = npy_path.parent.parent
            snapshot_dir = npy_path.parent
            if not npy_path.exists():
                raise FileNotFoundError(npy_path)

        else:
            exp_dir = self.data_root / experiment_name
            snapshot_dirs = sorted(
                exp_dir.glob("snapshot_*"),
                key=lambda p: p.stat().st_mtime,
            )
            if not snapshot_dirs:
                raise FileNotFoundError("No snapshots found.")

            snapshot_dir = snapshot_dirs[-1]

            npy_files = list(snapshot_dir.glob("data_*.npy"))
            if not npy_files:
                raise FileNotFoundError("No .npy data file found.")

            npy_path = npy_files[0]

        # -------- Load & plot --------
        img = np.load(npy_path)
        timestamp = snapshot_dir.name.replace("snapshot_", "")

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=0,
            vmax=self.max_value,
            origin="upper",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Intensity (counts, {self.bit_depth}-bit)", fontsize=10)
        cbar.set_ticks(np.linspace(0, self.max_value, 5).round().astype(int))
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        ax.set_title(f"{exp_dir.name}\n{timestamp}", fontsize=10)
        fig.tight_layout()
        plt.show()


