"""
ctypes bindings for the Thorlabs Scientific Camera SDK (thorlabs_tsi_camera_sdk.dll).

The DLL ships with ThorImageCAM and ThorCam. Thorlabs' own Python package
(thorlabs_tsi_sdk) is a ctypes wrapper of the same DLL that is not on PyPI, so
the few functions the Zelux driver needs are bound here instead. Every
function returns 0 on success; tl_camera_get_last_error() says what failed.

The SDK is opened once per process however many cameras use it
(open_sdk / close_sdk count the users).
"""

import ctypes as C
import os
from pathlib import Path

DLL_NAME = "thorlabs_tsi_camera_sdk.dll"

# Where the Thorlabs installers put the 64-bit SDK DLLs
_DLL_DIRS = (
    r"C:\Program Files\Thorlabs\ThorImageCAM\Bin",
    r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam",
    r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
    r"\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_64_lib",
)

# enum values of tl_camera_sdk.h
OPERATION_MODE_SOFTWARE_TRIGGERED = 0
SENSOR_TYPES = {0: "monochrome", 1: "bayer", 2: "monochrome polarized"}

STRING_MAX = 4096

_H = C.c_void_p
_PI = C.POINTER(C.c_int)
_PD = C.POINTER(C.c_double)
_PLL = C.POINTER(C.c_longlong)

_SIGNATURES = {
    "tl_camera_open_sdk": [],
    "tl_camera_close_sdk": [],
    "tl_camera_discover_available_cameras": [C.c_char_p, C.c_int],  # space-separated serial numbers
    "tl_camera_open_camera": [C.c_char_p, C.POINTER(C.c_void_p)],
    "tl_camera_close_camera": [_H],
    "tl_camera_get_serial_number": [_H, C.c_char_p, C.c_int],
    "tl_camera_get_model": [_H, C.c_char_p, C.c_int],
    "tl_camera_get_firmware_version": [_H, C.c_char_p, C.c_int],
    "tl_camera_get_camera_sensor_type": [_H, _PI],
    "tl_camera_get_sensor_width": [_H, _PI],
    "tl_camera_get_sensor_height": [_H, _PI],
    "tl_camera_get_sensor_pixel_width": [_H, _PD],  # um
    "tl_camera_get_bit_depth": [_H, _PI],
    "tl_camera_get_image_width": [_H, _PI],
    "tl_camera_get_image_height": [_H, _PI],
    "tl_camera_set_binx": [_H, C.c_int],
    "tl_camera_set_biny": [_H, C.c_int],
    "tl_camera_get_roi": [_H, _PI, _PI, _PI, _PI],  # upper-left x, y, lower-right x, y
    "tl_camera_set_roi": [_H, C.c_int, C.c_int, C.c_int, C.c_int],
    "tl_camera_get_roi_range": [_H] + [_PI] * 8,     # the four corners' minima, then their maxima
    "tl_camera_get_exposure_time": [_H, _PLL],       # us
    "tl_camera_set_exposure_time": [_H, C.c_longlong],
    "tl_camera_get_exposure_time_range": [_H, _PLL, _PLL],
    "tl_camera_get_gain": [_H, _PI],                 # gain index, not dB
    "tl_camera_set_gain": [_H, C.c_int],
    "tl_camera_get_gain_range": [_H, _PI, _PI],
    "tl_camera_convert_gain_to_decibels": [_H, C.c_int, _PD],
    "tl_camera_convert_decibels_to_gain": [_H, C.c_double, _PI],
    "tl_camera_get_black_level": [_H, _PI],
    "tl_camera_set_black_level": [_H, C.c_int],
    "tl_camera_get_black_level_range": [_H, _PI, _PI],
    "tl_camera_set_operation_mode": [_H, C.c_int],
    "tl_camera_set_frames_per_trigger_zero_for_unlimited": [_H, C.c_uint],
    "tl_camera_set_image_poll_timeout": [_H, C.c_int],  # ms
    "tl_camera_arm": [_H, C.c_int],                     # frames the SDK buffers
    "tl_camera_disarm": [_H],
    "tl_camera_issue_software_trigger": [_H],
    "tl_camera_get_pending_frame_or_null": [
        _H, C.POINTER(C.POINTER(C.c_ushort)), _PI, C.POINTER(C.POINTER(C.c_ubyte)), _PI,
    ],
}

_dll = None
_dll_path = None
_sdk_users = 0


def _candidate_paths(dll_path) -> list[Path]:
    """Explicit path first, then $THORLABS_TSI_SDK_DLL, then the Thorlabs install folders."""
    if dll_path is not None:
        return [Path(dll_path)]
    env = os.environ.get("THORLABS_TSI_SDK_DLL")
    return ([Path(env)] if env else []) + [Path(folder) / DLL_NAME for folder in _DLL_DIRS]


def load_api(dll_path=None) -> tuple[C.CDLL, str]:
    """Load and bind the SDK DLL (once per process). Returns the DLL and its path."""
    global _dll, _dll_path
    if _dll is not None:
        return _dll, _dll_path

    tried = []
    for path in _candidate_paths(dll_path):
        if not path.is_file():
            tried.append(f"{path}: not found")
            continue
        # The SDK loads its camera modules (thorlabs_tsi_zelux_camera_device.dll, ...) from its own folder
        folder = str(path.parent)
        os.add_dll_directory(folder)
        if folder not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ["PATH"]
        try:
            dll = C.CDLL(str(path))
        except OSError as err:
            tried.append(f"{path}: {err}")
            continue
        for name, argtypes in _SIGNATURES.items():
            func = getattr(dll, name)
            func.argtypes = argtypes
            func.restype = C.c_int
        dll.tl_camera_get_last_error.argtypes = []
        dll.tl_camera_get_last_error.restype = C.c_char_p
        _dll, _dll_path = dll, str(path)
        return _dll, _dll_path

    raise RuntimeError(
        "[Zelux] Thorlabs camera SDK not found. Tried:\n  " + "\n  ".join(tried)
        + "\nInstall ThorImageCAM or ThorCam from thorlabs.com, or point the THORLABS_TSI_SDK_DLL "
          f"environment variable at {DLL_NAME}."
    )


def last_error() -> str:
    text = _dll.tl_camera_get_last_error() if _dll is not None else None
    return text.decode(errors="replace").strip() if text else ""


def check(ret: int, what: str):
    if ret != 0:
        raise RuntimeError(f"[Zelux] {what} failed (error {ret}): {last_error() or 'no message'}")


def open_sdk():
    global _sdk_users
    if _sdk_users == 0:
        check(_dll.tl_camera_open_sdk(), "tl_camera_open_sdk")
    _sdk_users += 1


def close_sdk():
    global _sdk_users
    _sdk_users -= 1
    if _sdk_users == 0:
        _dll.tl_camera_close_sdk()


def discover() -> list[str]:
    """Serial numbers of the cameras the SDK sees."""
    buf = C.create_string_buffer(STRING_MAX)
    check(_dll.tl_camera_discover_available_cameras(buf, STRING_MAX), "tl_camera_discover_available_cameras")
    return buf.value.decode().split()
