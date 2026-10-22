"""
Cinogy CinCam CMOS 1201 beam profiler driven through the Thorlabs uc480 driver.

The CinCam CMOS 1201 is an IDS uEye UI-154xLE-M (1280 x 1024 mono CMOS,
5.2 um pixels) inside. It is a vendor-class USB device, not a UVC webcam, so
the Windows camera stack (DirectShow / Media Foundation / OpenCV) cannot see
it. Instead of the IDS driver bundled with CINOGY / RayCi, this class talks to
the free Thorlabs "uc480" driver (an OEM build of the same uEye API) through
ctypes. No RayCi, no pyueye.

Same snapshot interface as cinogy.CinCamCMOS1201 and allied_vision.AlviumG1:
- camera configuration is slow and happens once
- snapshots are fast and never query hardware
- all camera metadata is cached internally

Plus a live mode for viewers (camera_live_view):
start_live() / grab_latest() / stop_live(), set_aoi() / reset_aoi(), and the
settable ranges exposure_range_us() / gain_range_db().
"""

import ctypes as C
import json
import math

import numpy as np
from pathlib import Path
from datetime import datetime

from . import _uc480_api as api


class CinCamUC480:
    """
    Snapshot-oriented camera class.
    """

    # ---------- Pixel format ----------
    # name -> (uEye colour mode, bits per pixel in memory, bit depth of the values)
    _MONO_FORMATS = {
        "Mono12": (api.IS_CM_MONO12, 16, 12),
        "Mono10": (api.IS_CM_MONO10, 16, 10),
        "Mono8": (api.IS_CM_MONO8, 8, 8),
    }

    # Frames after opening that still carry the power-on settings (measured: 2)
    _STARTUP_STALE_FRAMES = 2

    # ---------- Declarative parameter registry ----------
    # Order matters: the pixel clock bounds the frame time, which bounds the exposure.
    _PARAMETERS = {
        "pixel_clock_mhz": {
            "default": None,  # None = keep the camera's default
            "setter": "set_pixel_clock",
        },
        "black_level": {
            "default": None,  # None = keep the camera's default (auto black level)
            "setter": "set_black_level",
        },
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
                uEye camera ID; None opens the first camera.
            dll_path : str or None
                Path to uc480_64.dll (or ueye_api_64.dll). None searches the
                CINCAM_UC480_DLL environment variable, PATH and the usual
                Thorlabs install folders.
            exposure_us : float or None
            gain_db : float or None
            exposure_auto : bool or None
            gain_auto : bool or None
            pixel_format : str or None
                "auto" (highest Mono bit depth the camera supports) or an
                explicit format: "Mono8", "Mono10", "Mono12".
            pixel_clock_mhz : int or None
                Sensor pixel clock. Lower clocks allow longer exposures.
            black_level : int or None
                Fixed black-level offset (0-255); turns off auto black level.
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
        self._live_bufs = []  # (pointer, id) of the live ring buffers; empty = not live
        self._frame_rate_fps = None
        self._stale_frame = True  # next frame was exposed with old settings

        self._dll, self.dll_path = api.load_api(self._cfg.get("dll_path", None))

        try:
            self._connect_camera()

            # Apply user configuration (sets camera state)
            self._apply_initial_settings()

            # ---- Metadata cache (must come LAST) ----
            self._camera_metadata = self._query_static_metadata()

            # The first frames after opening are still exposed with the camera's
            # power-on settings (~100 ms exposure), not ours: flush them.
            for _ in range(self._STARTUP_STALE_FRAMES):
                self._freeze()
            self._stale_frame = False
        except Exception:
            # Release the camera before re-raising, otherwise nothing else can open it
            try:
                self.close()
            except Exception:
                pass
            raise

    # ---------- Camera connection ----------

    def _check(self, ret: int, what: str):
        if ret != api.IS_SUCCESS:
            msg = ""
            if hasattr(self, "h_cam"):
                code, text = C.c_int(), C.c_char_p()
                self._dll.is_GetError(self.h_cam, C.byref(code), C.byref(text))
                msg = f": {text.value.decode(errors='replace')}" if text.value else ""
            raise RuntimeError(f"[CinCam uc480] {what} failed (uEye error code {ret}){msg}")

    def _connect_camera(self):
        """
        Connect to the CinCam through the uc480 driver.
        """
        h_cam = api.HIDS(self.camera_id or 0)
        ret = self._dll.is_InitCamera(C.byref(h_cam), None)
        if ret != api.IS_SUCCESS:
            raise RuntimeError(
                f"[CinCam uc480] Could not open the camera (uEye error code {ret}).\n"
                f"  RayCi, ThorCam or another script likely has it open.\n"
                f"  Close that application and run again."
            )
        self.h_cam = h_cam

        cam_info = api.CAMINFO()
        sensor_info = api.SENSORINFO()
        self._check(self._dll.is_GetCameraInfo(self.h_cam, C.byref(cam_info)), "is_GetCameraInfo")
        self._check(self._dll.is_GetSensorInfo(self.h_cam, C.byref(sensor_info)), "is_GetSensorInfo")

        self.camera_id = int(cam_info.Select)
        self.serial = cam_info.SerNo.decode()
        self.sensor_name = sensor_info.strSensorName.decode()
        self.sensor_width = int(sensor_info.nMaxWidth)
        self.sensor_height = int(sensor_info.nMaxHeight)
        self.pixel_size_um = int(sensor_info.wPixelSize) / 100  # stored in 0.01 um
        self._has_master_gain = bool(sensor_info.bMasterGain)

        version = self._dll.is_GetDLLVersion()
        self.dll_version = f"{(version >> 24) & 0xFF}.{(version >> 16) & 0xFF}.{version & 0xFFFF}"

        # Frames are copied into our own memory, not drawn to a window
        self._check(self._dll.is_SetDisplayMode(self.h_cam, api.IS_SET_DM_DIB), "is_SetDisplayMode")
        print(
            f"[CinCam uc480] Connected: {self.sensor_name} (serial {self.serial}) "
            f"via {Path(self.dll_path).name} {self.dll_version}"
        )

    def close(self):
        try:
            if hasattr(self, "h_cam"):
                self.stop_live()
            self._free_image_mem()
        finally:
            if hasattr(self, "h_cam"):
                self._dll.is_ExitCamera(self.h_cam)
                del self.h_cam

    # ---------- Image memory ----------

    def _get_aoi(self) -> api.IS_RECT:
        rect = api.IS_RECT()
        self._check(
            self._dll.is_AOI(self.h_cam, api.IS_AOI_IMAGE_GET_AOI, C.byref(rect), C.sizeof(rect)),
            "is_AOI(get)",
        )
        return rect

    def _alloc_image_mem(self):
        """
        (Re)allocate the frame buffer for the current AOI and pixel format.
        """
        self._free_image_mem()

        rect = self._get_aoi()
        self._width = int(rect.s32Width)
        self._height = int(rect.s32Height)

        mem_ptr = C.c_void_p()
        mem_id = C.c_int()
        self._check(
            self._dll.is_AllocImageMem(
                self.h_cam, self._width, self._height, self._bits_per_pixel,
                C.byref(mem_ptr), C.byref(mem_id),
            ),
            "is_AllocImageMem",
        )
        self._mem_ptr, self._mem_id = mem_ptr, mem_id
        self._check(self._dll.is_SetImageMem(self.h_cam, self._mem_ptr, self._mem_id), "is_SetImageMem")

        pitch = C.c_int()
        self._check(
            self._dll.is_InquireImageMem(
                self.h_cam, self._mem_ptr, self._mem_id, None, None, None, C.byref(pitch),
            ),
            "is_InquireImageMem",
        )
        self._pitch = int(pitch.value)

    def _free_image_mem(self):
        if self._mem_ptr is not None and hasattr(self, "h_cam"):
            self._dll.is_FreeImageMem(self.h_cam, self._mem_ptr, self._mem_id)
        self._mem_ptr = None

    # ---------- Declarative configuration ----------

    # ---------- Area of interest ----------

    @property
    def aoi(self) -> dict:
        """Readout rectangle in sensor pixels: offset_x, offset_y, width, height."""
        rect = self._get_aoi()
        return dict(offset_x=int(rect.s32X), offset_y=int(rect.s32Y),
                    width=int(rect.s32Width), height=int(rect.s32Height))

    def _apply_aoi(self, x: int, y: int, width: int, height: int):
        """Set the readout rectangle, pausing a running live stream for it."""
        was_live = self.live
        self.stop_live()
        rect = api.IS_RECT(x, y, width, height)
        self._check(
            self._dll.is_AOI(self.h_cam, api.IS_AOI_IMAGE_SET_AOI, C.byref(rect), C.sizeof(rect)),
            "is_AOI(set)",
        )
        self._alloc_image_mem()
        self._camera_metadata["aoi"] = self.aoi
        self._stale_frame = True
        # The frame-time range depends on the AOI and may have clamped the exposure
        exposure_us = self._camera_metadata.get("exposure_us")
        if exposure_us is not None and not self._camera_metadata.get("exposure_auto"):
            self.set_exposure(exposure_us)
        if was_live:
            self.start_live()

    def reset_aoi(self):
        """Read out the full sensor."""
        self._apply_aoi(0, 0, self.sensor_width, self.sensor_height)

    def set_aoi(
            self,
            offset_px: tuple[int, int] = (0, 0),
            width_px: int = None,
            height_px: int = None,
    ):
        """
        Restrict the sensor readout to a rectangle. offset_px is the top-left
        corner (column, row), with the origin at the sensor's top-left and y
        down. The rectangle is clamped to the sensor and aligned to the
        sensor's position / size increments: read `aoi` for what was set.
        """
        if width_px is None or height_px is None:
            raise ValueError("set_aoi() needs width_px and height_px.")

        size_inc = api.IS_SIZE_2D()
        pos_inc = api.IS_POINT_2D()
        size_min = api.IS_SIZE_2D()
        for cmd, struct, what in (
            (api.IS_AOI_IMAGE_GET_SIZE_INC, size_inc, "is_AOI(size inc)"),
            (api.IS_AOI_IMAGE_GET_POS_INC, pos_inc, "is_AOI(pos inc)"),
            (api.IS_AOI_IMAGE_GET_SIZE_MIN, size_min, "is_AOI(size min)"),
        ):
            self._check(self._dll.is_AOI(self.h_cam, cmd, C.byref(struct), C.sizeof(struct)), what)
        w_inc, h_inc = int(size_inc.s32Width), int(size_inc.s32Height)
        x_inc, y_inc = int(pos_inc.s32X), int(pos_inc.s32Y)
        w_min, h_min = int(size_min.s32Width), int(size_min.s32Height)

        # Clamp offset and size to the sensor, then align to the increments
        ox = (int(np.clip(offset_px[0], 0, self.sensor_width - w_min)) // x_inc) * x_inc
        oy = (int(np.clip(offset_px[1], 0, self.sensor_height - h_min)) // y_inc) * y_inc
        w = (int(np.clip(width_px, w_min, self.sensor_width - ox)) // w_inc) * w_inc
        h = (int(np.clip(height_px, h_min, self.sensor_height - oy)) // h_inc) * h_inc

        self._apply_aoi(ox, oy, w, h)
        print(f"[CinCam uc480] AOI set: offset=({ox}, {oy}), size=({w}x{h})")

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
        metadata = {
            "camera_model": "CinCam CMOS 1201",
            "driver": f"{Path(self.dll_path).name} {self.dll_version}",
            "sensor": self.sensor_name,
            "serial": self.serial,
            "camera_id": self.camera_id,
            "pixel_format": self._camera_metadata.get("pixel_format"),
            "bit_depth": self.bit_depth,
            "pixel_size_um": self.pixel_size_um,

            "resolution": [self.sensor_height, self.sensor_width],
            "aoi": self.aoi,
            "dead_pixel_file": str(self.dead_pixel_file) if self.dead_pixel_file.exists() else None,

            "exposure_us": self._get_exposure_us(),
            "gain_db": self._get_gain_db(),
            "exposure_auto": self._get_auto(api.IS_GET_ENABLE_AUTO_SHUTTER),
            "gain_auto": self._get_auto(api.IS_GET_ENABLE_AUTO_GAIN),
            "frame_rate_fps": self._frame_rate_fps,
            "pixel_clock_mhz": self._get_pixel_clock_mhz(),
            "black_level_auto": self._get_black_level_auto(),
            "black_level": self._get_black_level(),
        }
        return metadata

    def _get_exposure_us(self) -> float:
        exposure_ms = C.c_double()
        self._check(
            self._dll.is_Exposure(
                self.h_cam, api.IS_EXPOSURE_CMD_GET_EXPOSURE, C.byref(exposure_ms), C.sizeof(exposure_ms)
            ),
            "is_Exposure(get)",
        )
        return float(exposure_ms.value) * 1000

    def _get_gain_db(self) -> float:
        factor = self._dll.is_SetHWGainFactor(self.h_cam, api.IS_GET_MASTER_GAIN_FACTOR, 0)
        return 20 * math.log10(factor / 100)  # factor 100 = 1x

    def _get_auto(self, param) -> bool:
        enabled = C.c_double()
        self._check(
            self._dll.is_SetAutoParameter(self.h_cam, param, C.byref(enabled), C.byref(C.c_double())),
            "is_SetAutoParameter(get)",
        )
        return bool(enabled.value)

    def _get_pixel_clock_mhz(self) -> int:
        clock = C.c_uint()
        self._check(
            self._dll.is_PixelClock(self.h_cam, api.IS_PIXELCLOCK_CMD_GET, C.byref(clock), C.sizeof(clock)),
            "is_PixelClock(get)",
        )
        return int(clock.value)

    def _get_black_level_auto(self) -> bool:
        mode = C.c_int()
        self._check(
            self._dll.is_Blacklevel(self.h_cam, api.IS_BLACKLEVEL_CMD_GET_MODE, C.byref(mode), C.sizeof(mode)),
            "is_Blacklevel(get mode)",
        )
        return mode.value == api.IS_AUTO_BLACKLEVEL_ON

    def _get_black_level(self) -> int:
        offset = C.c_int()
        self._check(
            self._dll.is_Blacklevel(self.h_cam, api.IS_BLACKLEVEL_CMD_GET_OFFSET, C.byref(offset), C.sizeof(offset)),
            "is_Blacklevel(get offset)",
        )
        return int(offset.value)

    # ---------- Settable ranges and current values (for viewers) ----------

    @property
    def exposure_us(self) -> float:
        """Exposure actually in force (us), from the cache."""
        return self._camera_metadata["exposure_us"]

    @property
    def gain_db(self) -> float:
        return self._camera_metadata["gain_db"]

    def exposure_range_us(self) -> tuple[float, float]:
        """Settable exposure (us): the sensor minimum to the longest frame time."""
        exp_range = (C.c_double * 3)()  # min, max, increment (ms) at the current frame rate
        self._check(
            self._dll.is_Exposure(
                self.h_cam, api.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE, exp_range, C.sizeof(exp_range)
            ),
            "is_Exposure(range)",
        )
        t_min, t_max, t_inc = C.c_double(), C.c_double(), C.c_double()
        self._check(
            self._dll.is_GetFrameTimeRange(self.h_cam, C.byref(t_min), C.byref(t_max), C.byref(t_inc)),
            "is_GetFrameTimeRange",
        )
        return float(exp_range[0]) * 1000, float(t_max.value) * 1e6

    def gain_range_db(self) -> tuple[float, float]:
        if not self._has_master_gain:
            return 0.0, 0.0
        max_factor = self._dll.is_SetHWGainFactor(self.h_cam, api.IS_INQUIRE_MASTER_GAIN_FACTOR, 100)
        return 0.0, 20 * math.log10(max_factor / 100)

    # ---------- Parameter wrappers (update cache) ----------

    def set_exposure(self, exposure_us: float):
        """
        Set camera exposure time (microseconds).

        The maximum exposure is bounded by the frame time, so the frame rate is
        lowered first to make room for long exposures.
        """
        self.set_exposure_auto(False)

        t_min, t_max, t_inc = C.c_double(), C.c_double(), C.c_double()
        self._check(
            self._dll.is_GetFrameTimeRange(self.h_cam, C.byref(t_min), C.byref(t_max), C.byref(t_inc)),
            "is_GetFrameTimeRange",
        )
        frame_time_s = float(np.clip(exposure_us * 1e-6, t_min.value, t_max.value))

        new_fps = C.c_double()
        self._check(self._dll.is_SetFrameRate(self.h_cam, 1 / frame_time_s, C.byref(new_fps)), "is_SetFrameRate")
        self._frame_rate_fps = float(new_fps.value)

        exposure_ms = C.c_double(exposure_us / 1000)
        self._check(
            self._dll.is_Exposure(
                self.h_cam, api.IS_EXPOSURE_CMD_SET_EXPOSURE, C.byref(exposure_ms), C.sizeof(exposure_ms)
            ),
            "is_Exposure(set)",
        )

        # The sensor rounds to its line time; record what was actually set
        actual_us = self._get_exposure_us()
        if abs(actual_us - exposure_us) > max(0.01 * exposure_us, 50):
            print(f"[CinCam uc480] WARNING: requested {exposure_us} us exposure, camera set {actual_us:.1f} us")

        self._camera_metadata["exposure_us"] = actual_us
        self._camera_metadata["exposure_auto"] = False
        self._camera_metadata["frame_rate_fps"] = self._frame_rate_fps
        self._stale_frame = True

    def set_gain(self, gain_db: float):
        """
        Set camera master gain (dB). uEye gain is a factor where 100 = 1x.
        """
        if not self._has_master_gain:
            print("[CinCam uc480] Sensor has no master gain, ignoring gain setting.")
            return

        self.set_gain_auto(False)
        # Inquiring with factor 100 returns the maximum factor (other values return -1)
        max_factor = self._dll.is_SetHWGainFactor(self.h_cam, api.IS_INQUIRE_MASTER_GAIN_FACTOR, 100)
        factor = int(np.clip(round(100 * 10 ** (gain_db / 20)), 100, max_factor))
        self._dll.is_SetHWGainFactor(self.h_cam, api.IS_SET_MASTER_GAIN_FACTOR, factor)

        actual_db = self._get_gain_db()
        if abs(actual_db - gain_db) > 0.1:
            print(f"[CinCam uc480] WARNING: requested {gain_db} dB gain, camera set {actual_db:.2f} dB")

        self._camera_metadata["gain_db"] = actual_db
        self._camera_metadata["gain_auto"] = False
        self._stale_frame = True

    def set_exposure_auto(self, enabled: bool = True):
        """
        Enable or disable automatic exposure.
        """
        self._check(
            self._dll.is_SetAutoParameter(
                self.h_cam, api.IS_SET_ENABLE_AUTO_SHUTTER, C.byref(C.c_double(enabled)), C.byref(C.c_double())
            ),
            "is_SetAutoParameter(auto exposure)",
        )
        self._camera_metadata["exposure_auto"] = enabled
        self._stale_frame = True

    def set_gain_auto(self, enabled: bool = True):
        """
        Enable or disable automatic gain.
        """
        self._check(
            self._dll.is_SetAutoParameter(
                self.h_cam, api.IS_SET_ENABLE_AUTO_GAIN, C.byref(C.c_double(enabled)), C.byref(C.c_double())
            ),
            "is_SetAutoParameter(auto gain)",
        )
        self._camera_metadata["gain_auto"] = enabled
        self._stale_frame = True

    def set_pixel_clock(self, pixel_clock_mhz: int):
        """
        Set the sensor pixel clock (MHz). This changes the allowed frame time
        range, so set it before the exposure.
        """
        clock_range = (C.c_uint * 3)()  # min, max, increment
        self._check(
            self._dll.is_PixelClock(
                self.h_cam, api.IS_PIXELCLOCK_CMD_GET_RANGE, clock_range, C.sizeof(clock_range)
            ),
            "is_PixelClock(range)",
        )
        c_min, c_max, _ = clock_range
        clock = C.c_uint(int(np.clip(pixel_clock_mhz, c_min, c_max)))
        self._check(
            self._dll.is_PixelClock(self.h_cam, api.IS_PIXELCLOCK_CMD_SET, C.byref(clock), C.sizeof(clock)),
            "is_PixelClock(set)",
        )
        if clock.value != pixel_clock_mhz:
            print(f"[CinCam uc480] WARNING: pixel clock {pixel_clock_mhz} MHz out of range, set {clock.value} MHz")

        self._camera_metadata["pixel_clock_mhz"] = int(clock.value)
        self._stale_frame = True

    def set_black_level(self, offset: int):
        """
        Fix the black-level offset (counts) and turn off auto black level, so
        the dark level stays the same between snapshots.
        """
        mode = C.c_int(api.IS_AUTO_BLACKLEVEL_OFF)
        self._check(
            self._dll.is_Blacklevel(self.h_cam, api.IS_BLACKLEVEL_CMD_SET_MODE, C.byref(mode), C.sizeof(mode)),
            "is_Blacklevel(set mode)",
        )
        value = C.c_int(int(offset))
        self._check(
            self._dll.is_Blacklevel(self.h_cam, api.IS_BLACKLEVEL_CMD_SET_OFFSET, C.byref(value), C.sizeof(value)),
            "is_Blacklevel(set offset)",
        )
        self._camera_metadata["black_level_auto"] = False
        self._camera_metadata["black_level"] = int(offset)
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
            if self._dll.is_SetColorMode(self.h_cam, mode) == api.IS_SUCCESS:
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
        print(f"[CinCam uc480] Pixel format: {name} ({self.bit_depth}-bit)")

    # ---------- Image acquisition ----------

    def _read_buffer(self, address: int) -> np.ndarray:
        """Copy the frame in the driver buffer at `address` into a uint16 array."""
        buf = (C.c_ubyte * (self._pitch * self._height)).from_address(address)
        dtype = np.uint16 if self._bits_per_pixel == 16 else np.uint8
        rows = np.frombuffer(buf, dtype=np.uint8).reshape(self._height, self._pitch).view(dtype)
        return rows[:, : self._width].astype(np.uint16)  # astype copies out of the driver's buffer

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

        # The rolling-shutter sensor applies new settings one frame late, so the
        # first frame after any change still has the old exposure: discard it.
        if self._stale_frame:
            self._freeze()
            self._stale_frame = False
        self._freeze()
        return self._read_buffer(self._mem_ptr.value)

    def _freeze(self):
        """One software-triggered capture into the snapshot buffer."""
        ret = self._dll.is_FreezeVideo(self.h_cam, api.IS_WAIT)
        if ret == api.IS_TIMED_OUT:
            # Right after an AOI change the first request fails at once with a
            # timeout while the sensor reconfigures; the next one succeeds.
            ret = self._dll.is_FreezeVideo(self.h_cam, api.IS_WAIT)
        self._check(ret, "is_FreezeVideo")

    # ---------- Live acquisition ----------
    # Free-running mode for viewers. The driver cycles through a ring of its
    # own buffers (is_AddToSequence) and queues every finished frame
    # (is_InitImageQueue); grab_latest() hands out the newest one and drops
    # the rest, without ever blocking. Measured on the CinCam: 14 fps full
    # frame, 155 fps on a 300 x 200 AOI; a setting change shows two frames
    # later; an AOI change (stop / restart) takes ~85 ms.

    _LIVE_BUFFERS = 4

    @property
    def live(self) -> bool:
        return bool(self._live_bufs)

    def start_live(self):
        """Start free-running capture into the live buffers (no-op if running)."""
        if self.live:
            return
        for _ in range(self._LIVE_BUFFERS):
            ptr, mem_id = C.c_void_p(), C.c_int()
            self._check(
                self._dll.is_AllocImageMem(
                    self.h_cam, self._width, self._height, self._bits_per_pixel,
                    C.byref(ptr), C.byref(mem_id),
                ),
                "is_AllocImageMem",
            )
            self._check(self._dll.is_AddToSequence(self.h_cam, ptr, mem_id), "is_AddToSequence")
            self._live_bufs.append((ptr, mem_id))
        self._check(self._dll.is_InitImageQueue(self.h_cam, 0), "is_InitImageQueue")
        self._check(self._dll.is_CaptureVideo(self.h_cam, api.IS_DONT_WAIT), "is_CaptureVideo")

    def stop_live(self):
        """Stop the stream and free the live buffers (no-op if not running)."""
        if not self.live:
            return
        self._dll.is_StopLiveVideo(self.h_cam, api.IS_FORCE_VIDEO_STOP)
        self._dll.is_ExitImageQueue(self.h_cam)
        self._dll.is_ClearSequence(self.h_cam)
        for ptr, mem_id in self._live_bufs:
            self._dll.is_FreeImageMem(self.h_cam, ptr, mem_id)
        self._live_bufs = []
        # The sequence replaced the snapshot buffer as the active image memory
        self._check(self._dll.is_SetImageMem(self.h_cam, self._mem_ptr, self._mem_id), "is_SetImageMem")
        self._stale_frame = True

    def grab_latest(self) -> np.ndarray | None:
        """
        The newest live frame, or None if none finished since the last call.
        Never blocks; frames that queued up in between are skipped.
        """
        ptr, mem_id = C.c_void_p(), C.c_int()
        newest = None
        while self._dll.is_WaitForNextImage(self.h_cam, 0, C.byref(ptr), C.byref(mem_id)) == api.IS_SUCCESS:
            if newest is not None:
                self._dll.is_UnlockSeqBuf(self.h_cam, *newest)
            newest = (mem_id.value, ptr.value)
        if newest is None:
            return None
        frame = self._read_buffer(newest[1])
        self._dll.is_UnlockSeqBuf(self.h_cam, *newest)
        return frame

    def frame_rate(self) -> float:
        """Frames per second the camera currently delivers (live mode)."""
        fps = C.c_double()
        self._dll.is_GetFramesPerSecond(self.h_cam, C.byref(fps))
        return float(fps.value)

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
            self.set_aoi(**cropping_params)
        try:
            img = self.capture_frame()
        finally:
            if cropping_params is not None:
                self.reset_aoi()

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

        print(f"[CinCam uc480] Snapshot saved to {base_dir}")

        return img, base_dir
