"""
Minimal camera abstraction for the Cinogy CinCam CMOS 1201 beam profiler.

The CinCam CMOS 1201 is an IDS uEye UI-154xLE-M (1280 x 1024 mono CMOS,
5.2 um pixels) inside, so it is driven directly through the IDS uEye API
(pyueye -> ueye_api_64.dll, installed with the CINOGY / RayCi driver).
RayCi does not need to be running; it must in fact be closed, since only
one program can hold the camera.

Mirrors the AlviumG1 snapshot interface:
- camera configuration is slow and happens once
- snapshots are fast and never query hardware
- all camera metadata is cached internally
"""

import json
import math

import numpy as np
from pathlib import Path
from datetime import datetime
from pyueye import ueye


class CinCamCMOS1201:
    """
    Snapshot-oriented camera class.
    """

    # ---------- Pixel format ----------
    # name -> (uEye colour mode, bits per pixel in memory, bit depth of the values)
    _MONO_FORMATS = {
        "Mono12": (ueye.IS_CM_MONO12, 16, 12),
        "Mono10": (ueye.IS_CM_MONO10, 16, 10),
        "Mono8": (ueye.IS_CM_MONO8, 8, 8),
    }

    # ---------- Declarative parameter registry ----------
    _PARAMETERS = {
        "exposure_auto": {
            "default": False,
            "setter": "set_exposure_auto",
        },
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
            camera_id : int or None
                uEye camera ID (set in IDS Camera Manager); None opens the first camera.
            exposure_us : float or None
            gain_db : float or None
            exposure_auto : bool or None
            gain_auto : bool or None
            pixel_format : str or None
                "auto" (highest Mono bit depth the camera supports) or an
                explicit format: "Mono8", "Mono10", "Mono12".
        """
        # Store config
        self._cfg = config

        # Data directory
        self.data_root = Path(self._cfg["data_root"])
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Camera ID (0 = first available camera)
        self.camera_id = self._cfg.get("camera_id", None)
        self._camera_metadata = {}
        self._mem_ptr = None
        self._frame_rate_fps = None
        self._stale_frame = True  # next frame was exposed with old settings

        try:
            self._connect_camera()

            # Apply user configuration (sets camera state)
            self._apply_initial_settings()

            # ---- Metadata cache (must come LAST) ----
            self._camera_metadata = self._query_static_metadata()
        except Exception:
            # Release the camera before re-raising, otherwise RayCi cannot reopen it
            try:
                self.close()
            except Exception:
                pass
            raise

    # ---------- Camera connection ----------

    @staticmethod
    def _check(ret: int, what: str):
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"[CinCam] {what} failed (uEye error code {ret})")

    def _connect_camera(self):
        """
        Connect to the CinCam (uEye) camera.
        """
        self.h_cam = ueye.HIDS(self.camera_id or 0)
        ret = ueye.is_InitCamera(self.h_cam, None)
        if ret != ueye.IS_SUCCESS:
            del self.h_cam  # nothing to close
            raise RuntimeError(
                f"[CinCam] Could not open the camera (uEye error code {ret}).\n"
                f"  RayCi (or IDS Camera Manager / uEye Cockpit) likely has it open.\n"
                f"  Close that application and run again."
            )

        cam_info = ueye.CAMINFO()
        sensor_info = ueye.SENSORINFO()
        self._check(ueye.is_GetCameraInfo(self.h_cam, cam_info), "is_GetCameraInfo")
        self._check(ueye.is_GetSensorInfo(self.h_cam, sensor_info), "is_GetSensorInfo")

        self.camera_id = int(cam_info.Select.value)
        self.serial = cam_info.SerNo.decode()
        self.sensor_name = sensor_info.strSensorName.decode()
        self.sensor_width = int(sensor_info.nMaxWidth)
        self.sensor_height = int(sensor_info.nMaxHeight)
        self.pixel_size_um = int(sensor_info.wPixelSize) / 100  # stored in 0.01 um
        self._has_master_gain = bool(sensor_info.bMasterGain)

        # Frames are copied into our own memory, not drawn to a window
        self._check(ueye.is_SetDisplayMode(self.h_cam, ueye.IS_SET_DM_DIB), "is_SetDisplayMode")
        print(f"[CinCam] Connected: {self.sensor_name} (serial {self.serial})")

    def close(self):
        try:
            self._free_image_mem()
        finally:
            if hasattr(self, "h_cam"):
                ueye.is_ExitCamera(self.h_cam)
                del self.h_cam

    # ---------- Image memory ----------

    def _alloc_image_mem(self):
        """
        (Re)allocate the frame buffer for the current AOI and pixel format.
        """
        self._free_image_mem()

        rect = ueye.IS_RECT()
        self._check(
            ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect)),
            "is_AOI(get)",
        )
        self._width = int(rect.s32Width)
        self._height = int(rect.s32Height)

        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()
        self._check(
            ueye.is_AllocImageMem(
                self.h_cam, self._width, self._height, self._bits_per_pixel,
                self._mem_ptr, self._mem_id,
            ),
            "is_AllocImageMem",
        )
        self._check(ueye.is_SetImageMem(self.h_cam, self._mem_ptr, self._mem_id), "is_SetImageMem")

        pitch = ueye.INT()
        self._check(
            ueye.is_InquireImageMem(
                self.h_cam, self._mem_ptr, self._mem_id,
                ueye.INT(), ueye.INT(), ueye.INT(), pitch,
            ),
            "is_InquireImageMem",
        )
        self._pitch = int(pitch.value)

    def _free_image_mem(self):
        if self._mem_ptr is not None and hasattr(self, "h_cam"):
            ueye.is_FreeImageMem(self.h_cam, self._mem_ptr, self._mem_id)
        self._mem_ptr = None

    # ---------- Declarative configuration ----------

    def _set_aoi(self, x: int, y: int, width: int, height: int):
        rect = ueye.IS_RECT()
        rect.s32X, rect.s32Y = ueye.int(x), ueye.int(y)
        rect.s32Width, rect.s32Height = ueye.int(width), ueye.int(height)
        self._check(
            ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect)),
            "is_AOI(set)",
        )
        self._alloc_image_mem()
        self._stale_frame = True

    def _uncrop_camera(self):
        self._set_aoi(0, 0, self.sensor_width, self.sensor_height)
        print("AOI:", self._width, self._height, 0, 0)

    def _crop_camera(
            self,
            offset_px: tuple[int, int] = (0, 0),
            width_px: int = None,
            height_px: int = None,
            ):
        """
        Restrict the sensor readout to a rectangle. offset_px is the top-left
        corner (column, row), with the origin at the sensor's top-left and y down.
        """
        if width_px is None or height_px is None:
            print("[CinCam] No cropping params given, skipping crop.")
            return

        size_inc = ueye.IS_SIZE_2D()
        pos_inc = ueye.IS_POINT_2D()
        size_min = ueye.IS_SIZE_2D()
        self._check(ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_SIZE_INC, size_inc, ueye.sizeof(size_inc)), "is_AOI(size inc)")
        self._check(ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_POS_INC, pos_inc, ueye.sizeof(pos_inc)), "is_AOI(pos inc)")
        self._check(ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_SIZE_MIN, size_min, ueye.sizeof(size_min)), "is_AOI(size min)")
        w_inc, h_inc = int(size_inc.s32Width), int(size_inc.s32Height)
        x_inc, y_inc = int(pos_inc.s32X), int(pos_inc.s32Y)
        w_min, h_min = int(size_min.s32Width), int(size_min.s32Height)

        # Clamp offset and size to the sensor, then align to the increments
        ox = (int(np.clip(offset_px[0], 0, self.sensor_width - w_min)) // x_inc) * x_inc
        oy = (int(np.clip(offset_px[1], 0, self.sensor_height - h_min)) // y_inc) * y_inc
        w = (int(np.clip(width_px, w_min, self.sensor_width - ox)) // w_inc) * w_inc
        h = (int(np.clip(height_px, h_min, self.sensor_height - oy)) // h_inc) * h_inc

        self._set_aoi(ox, oy, w, h)
        print(f"[CinCam] AOI set: offset=({ox}, {oy}), size=({w}x{h})")

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

        # Full sensor AOI; also allocates the frame buffer for the pixel format
        self._uncrop_camera()

    # ---------- Metadata cache ----------

    def _query_static_metadata(self) -> dict:
        """
        Query camera metadata once and return JSON-safe values.
        """
        metadata = {
            "camera_model": "CinCam CMOS 1201",
            "sensor": self.sensor_name,
            "serial": self.serial,
            "camera_id": self.camera_id,
            "pixel_format": self._camera_metadata.get("pixel_format"),
            "bit_depth": self.bit_depth,
            "pixel_size_um": self.pixel_size_um,

            "resolution": [self.sensor_height, self.sensor_width],

            "exposure_us": self._get_exposure_us(),
            "gain_db": self._get_gain_db(),
            "exposure_auto": self._get_auto(ueye.IS_GET_ENABLE_AUTO_SHUTTER),
            "gain_auto": self._get_auto(ueye.IS_GET_ENABLE_AUTO_GAIN),
            "frame_rate_fps": self._frame_rate_fps,
        }
        return metadata

    def _get_exposure_us(self) -> float:
        exposure_ms = ueye.DOUBLE()
        self._check(
            ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure_ms, ueye.sizeof(exposure_ms)),
            "is_Exposure(get)",
        )
        return float(exposure_ms.value) * 1000

    def _get_gain_db(self) -> float:
        factor = ueye.is_SetHWGainFactor(self.h_cam, ueye.IS_GET_MASTER_GAIN_FACTOR, 0)
        return 20 * math.log10(factor / 100)  # factor 100 = 1x

    def _get_auto(self, param) -> bool:
        enabled = ueye.DOUBLE()
        self._check(ueye.is_SetAutoParameter(self.h_cam, param, enabled, ueye.DOUBLE()), "is_SetAutoParameter(get)")
        return bool(enabled.value)

    # ---------- Parameter wrappers (update cache) ----------

    def set_exposure(self, exposure_us: float):
        """
        Set camera exposure time (microseconds).

        The maximum exposure is bounded by the frame time, so the frame rate is
        lowered first to make room for long exposures.
        """
        self.set_exposure_auto(False)

        t_min, t_max, t_inc = ueye.DOUBLE(), ueye.DOUBLE(), ueye.DOUBLE()
        self._check(ueye.is_GetFrameTimeRange(self.h_cam, t_min, t_max, t_inc), "is_GetFrameTimeRange")
        frame_time_s = float(np.clip(exposure_us * 1e-6, t_min.value, t_max.value))

        new_fps = ueye.DOUBLE()
        self._check(ueye.is_SetFrameRate(self.h_cam, ueye.DOUBLE(1 / frame_time_s), new_fps), "is_SetFrameRate")
        self._frame_rate_fps = float(new_fps.value)

        exposure_ms = ueye.DOUBLE(exposure_us / 1000)
        self._check(
            ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_ms, ueye.sizeof(exposure_ms)),
            "is_Exposure(set)",
        )

        # The sensor rounds to its line time; record what was actually set
        actual_us = self._get_exposure_us()
        if abs(actual_us - exposure_us) > max(0.01 * exposure_us, 50):
            print(f"[CinCam] WARNING: requested {exposure_us} us exposure, camera set {actual_us:.1f} us")

        self._camera_metadata["exposure_us"] = actual_us
        self._camera_metadata["exposure_auto"] = False
        self._camera_metadata["frame_rate_fps"] = self._frame_rate_fps
        self._stale_frame = True

    def set_gain(self, gain_db: float):
        """
        Set camera master gain (dB). uEye gain is a factor where 100 = 1x.
        """
        if not self._has_master_gain:
            print("[CinCam] Sensor has no master gain, ignoring gain setting.")
            return

        self.set_gain_auto(False)
        factor = round(100 * 10 ** (gain_db / 20))
        ueye.is_SetHWGainFactor(self.h_cam, ueye.IS_SET_MASTER_GAIN_FACTOR, factor)

        actual_db = self._get_gain_db()
        if abs(actual_db - gain_db) > 0.1:
            print(f"[CinCam] WARNING: requested {gain_db} dB gain, camera set {actual_db:.2f} dB")

        self._camera_metadata["gain_db"] = actual_db
        self._camera_metadata["gain_auto"] = False
        self._stale_frame = True

    def set_exposure_auto(self, enabled: bool = True):
        """
        Enable or disable automatic exposure.
        """
        self._check(
            ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.DOUBLE(enabled), ueye.DOUBLE()),
            "is_SetAutoParameter(auto exposure)",
        )
        self._camera_metadata["exposure_auto"] = enabled
        self._stale_frame = True

    def set_gain_auto(self, enabled: bool = True):
        """
        Enable or disable automatic gain.
        """
        self._check(
            ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.DOUBLE(enabled), ueye.DOUBLE()),
            "is_SetAutoParameter(auto gain)",
        )
        self._camera_metadata["gain_auto"] = enabled
        self._stale_frame = True

    def set_pixel_format(self, format="auto"):
        """
        Set camera pixel format.

        "auto" picks the highest Mono bit depth the camera accepts.
        An explicit name (e.g. "Mono10") must be supported by the camera.
        """
        if format == "auto":
            candidates = list(self._MONO_FORMATS)
        elif format in self._MONO_FORMATS:
            candidates = [format]
        else:
            raise ValueError(
                f"Pixel format '{format}' not supported. Choose from {list(self._MONO_FORMATS)}"
            )

        for name in candidates:
            mode, bits_per_pixel, bit_depth = self._MONO_FORMATS[name]
            if ueye.is_SetColorMode(self.h_cam, mode) == ueye.IS_SUCCESS:
                break
        else:
            raise RuntimeError(f"Camera accepted none of the pixel formats {candidates}")

        self._bits_per_pixel = bits_per_pixel
        self.bit_depth = bit_depth
        self.max_value = 2 ** self.bit_depth - 1
        if self._mem_ptr is not None:
            self._alloc_image_mem()  # buffer size depends on the format

        self._camera_metadata["pixel_format"] = name
        self._camera_metadata["bit_depth"] = self.bit_depth
        self._stale_frame = True
        print(f"[CinCam] Pixel format: {name} ({self.bit_depth}-bit)")

    # ---------- Image acquisition ----------

    def capture_frame(self) -> np.ndarray:
        """
        Acquire a single image frame (software-triggered single capture).

        Returns
        -------
        np.ndarray
            Raw camera image (uint16, values 0 to self.max_value).
        """
        # The rolling-shutter sensor applies new settings one frame late, so the
        # first frame after any change still has the old exposure: discard it.
        if self._stale_frame:
            self._check(ueye.is_FreezeVideo(self.h_cam, ueye.IS_WAIT), "is_FreezeVideo")
            self._stale_frame = False
        self._check(ueye.is_FreezeVideo(self.h_cam, ueye.IS_WAIT), "is_FreezeVideo")

        raw = ueye.get_data(
            self._mem_ptr, self._width, self._height, self._bits_per_pixel, self._pitch, copy=True
        )
        dtype = np.uint16 if self._bits_per_pixel == 16 else np.uint8
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._pitch).view(dtype)
        img = rows[:, : self._width]
        return img.astype(np.uint16)

    # ---------- Snapshot saving ----------

    def snapshot(
        self,
        experiment_name: str,
        comment: str = "",
        save_preview: bool = False,
        save_metadata : bool = True,
        path : str|Path = None,
        filename : str = None,
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

        if cropping_params is not None:
            self._crop_camera(**cropping_params)
        try:
            img = self.capture_frame()
        finally:
            if cropping_params is not None:
                self._uncrop_camera()

        if exposure_time_us is not None:
            exposure_us = self._cfg.get("exposure_us")
            if exposure_us is not None:
                default_value_us = exposure_us
            else:
                default_value_us = self._PARAMETERS["exposure_us"]["default"]
            self.set_exposure(exposure_us=default_value_us)
            print("set back exposure time us to default value", default_value_us)

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
            # Shared PSF preview plots live in the allied_vision package for now
            from allied_vision.camera_plot_helpers import save_psf_2d, save_psf_3d
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
        if save_metadata :
            metadata = {
                **self._camera_metadata,
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

        print(f"[CinCam] Snapshot saved to {base_dir}")

        return img, base_dir
