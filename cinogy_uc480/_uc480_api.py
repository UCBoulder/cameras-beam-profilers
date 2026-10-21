"""
Thin ctypes binding to the uEye C API, as shipped in the Thorlabs uc480 DLL.

Thorlabs (DCx cameras, wavefront sensors, ThorCam) rebrand IDS uEye cameras and
install the uEye driver under the name "uc480". The C API is identical to IDS's
ueye_api_64.dll: same function names, same constants. Only the functions this
package uses are bound here, with explicit argtypes so 64-bit pointers and
doubles are passed correctly.

Constants were taken from pyueye 4.96 (ueye.py).

Which DLL to load depends on which kernel driver Windows bound the camera to:
uc480_64.dll talks to the "uc480" service, ueye_api_64.dll to the IDS "ueye"
service. `load_api` tries every candidate and keeps the first that sees a camera.
"""

import ctypes as C
import os
from pathlib import Path

# ---------- Constants (from pyueye 4.96) ----------

IS_SUCCESS = 0
IS_TIMED_OUT = 122
IS_DONT_WAIT = 0x0000
IS_WAIT = 0x0001
IS_FORCE_VIDEO_STOP = 0x4000
IS_SET_DM_DIB = 1

IS_GET_COLOR_MODE = 0x8000
IS_CM_MONO8 = 6
IS_CM_MONO10 = 34
IS_CM_MONO12 = 26
IS_CM_MONO16 = 28

IS_AOI_IMAGE_SET_AOI = 0x0001
IS_AOI_IMAGE_GET_AOI = 0x0002
IS_AOI_IMAGE_GET_SIZE_MIN = 0x0008
IS_AOI_IMAGE_GET_POS_INC = 0x0011
IS_AOI_IMAGE_GET_SIZE_INC = 0x0012

IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE = 6
IS_EXPOSURE_CMD_GET_EXPOSURE = 7
IS_EXPOSURE_CMD_SET_EXPOSURE = 12

IS_GET_MASTER_GAIN_FACTOR = 0x8000
IS_SET_MASTER_GAIN_FACTOR = 0x8004
IS_INQUIRE_MASTER_GAIN_FACTOR = 0x800C

IS_SET_ENABLE_AUTO_GAIN = 0x8800
IS_GET_ENABLE_AUTO_GAIN = 0x8801
IS_SET_ENABLE_AUTO_SHUTTER = 0x8802
IS_GET_ENABLE_AUTO_SHUTTER = 0x8803

IS_PIXELCLOCK_CMD_GET_RANGE = 3
IS_PIXELCLOCK_CMD_GET = 5
IS_PIXELCLOCK_CMD_SET = 6

IS_BLACKLEVEL_CMD_GET_MODE = 3
IS_BLACKLEVEL_CMD_SET_MODE = 4
IS_BLACKLEVEL_CMD_GET_OFFSET_RANGE = 6
IS_BLACKLEVEL_CMD_GET_OFFSET = 7
IS_BLACKLEVEL_CMD_SET_OFFSET = 8
IS_AUTO_BLACKLEVEL_OFF = 0
IS_AUTO_BLACKLEVEL_ON = 1

# ---------- Structures ----------


class CAMINFO(C.Structure):
    _fields_ = [
        ("SerNo", C.c_char * 12),
        ("ID", C.c_char * 20),
        ("Version", C.c_char * 10),
        ("Date", C.c_char * 12),
        ("Select", C.c_ubyte),
        ("Type", C.c_ubyte),
        ("Reserved", C.c_char * 8),
    ]


class SENSORINFO(C.Structure):
    _fields_ = [
        ("SensorID", C.c_uint16),
        ("strSensorName", C.c_char * 32),
        ("nColorMode", C.c_char),
        ("nMaxWidth", C.c_uint32),
        ("nMaxHeight", C.c_uint32),
        ("bMasterGain", C.c_int32),
        ("bRGain", C.c_int32),
        ("bGGain", C.c_int32),
        ("bBGain", C.c_int32),
        ("bGlobShutter", C.c_int32),
        ("wPixelSize", C.c_uint16),  # in 0.01 um
        ("nUpperLeftBayerPixel", C.c_char),
        ("Reserved", C.c_char * 13),
    ]


class IS_RECT(C.Structure):
    _fields_ = [("s32X", C.c_int32), ("s32Y", C.c_int32), ("s32Width", C.c_int32), ("s32Height", C.c_int32)]


class IS_POINT_2D(C.Structure):
    _fields_ = [("s32X", C.c_int32), ("s32Y", C.c_int32)]


class IS_SIZE_2D(C.Structure):
    _fields_ = [("s32Width", C.c_int32), ("s32Height", C.c_int32)]


# ---------- Function signatures ----------

HIDS = C.c_uint32
_P = C.c_void_p
_PI = C.POINTER(C.c_int)
_PD = C.POINTER(C.c_double)

_SIGNATURES = {
    "is_GetDLLVersion": [],
    "is_GetNumberOfCameras": [_PI],
    "is_InitCamera": [C.POINTER(HIDS), _P],
    "is_ExitCamera": [HIDS],
    "is_GetError": [HIDS, _PI, C.POINTER(C.c_char_p)],
    "is_GetCameraInfo": [HIDS, C.POINTER(CAMINFO)],
    "is_GetSensorInfo": [HIDS, C.POINTER(SENSORINFO)],
    "is_SetDisplayMode": [HIDS, C.c_int],
    "is_SetColorMode": [HIDS, C.c_int],
    "is_AOI": [HIDS, C.c_uint, _P, C.c_uint],
    "is_AllocImageMem": [HIDS, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p), _PI],
    "is_SetImageMem": [HIDS, C.c_void_p, C.c_int],
    "is_FreeImageMem": [HIDS, C.c_void_p, C.c_int],
    "is_InquireImageMem": [HIDS, C.c_void_p, C.c_int, _PI, _PI, _PI, _PI],
    "is_FreezeVideo": [HIDS, C.c_int],
    # live mode: ring of buffers + frame queue
    "is_CaptureVideo": [HIDS, C.c_int],
    "is_StopLiveVideo": [HIDS, C.c_int],
    "is_AddToSequence": [HIDS, C.c_void_p, C.c_int],
    "is_ClearSequence": [HIDS],
    "is_InitImageQueue": [HIDS, C.c_int],
    "is_ExitImageQueue": [HIDS],
    "is_WaitForNextImage": [HIDS, C.c_uint, C.POINTER(C.c_void_p), _PI],
    "is_UnlockSeqBuf": [HIDS, C.c_int, C.c_void_p],
    "is_GetFramesPerSecond": [HIDS, _PD],
    "is_Exposure": [HIDS, C.c_uint, _P, C.c_uint],
    "is_GetFrameTimeRange": [HIDS, _PD, _PD, _PD],
    "is_SetFrameRate": [HIDS, C.c_double, _PD],
    "is_SetHWGainFactor": [HIDS, C.c_int, C.c_int],
    "is_SetAutoParameter": [HIDS, C.c_int, _PD, _PD],
    "is_PixelClock": [HIDS, C.c_uint, _P, C.c_uint],
    "is_Blacklevel": [HIDS, C.c_uint, _P, C.c_uint],
}

# ---------- DLL discovery ----------

_DLL_NAMES = ("uc480_64.dll", "ueye_api_64.dll")

_DLL_DIRS = (
    r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam",
    r"C:\Program Files (x86)\Thorlabs\Wavefront Sensor\Drivers\uc480\amd64",
    r"C:\Program Files (x86)\Thorlabs\Wavefront Sensor",
)


def _candidate_paths(dll_path: str | None) -> list[str]:
    """
    Explicit path first, then $CINCAM_UC480_DLL, then bare names (System32 /
    PATH), then known vendor install folders.
    """
    if dll_path is not None:
        return [str(dll_path)]

    candidates = []
    env = os.environ.get("CINCAM_UC480_DLL")
    if env:
        candidates.append(env)
    candidates.extend(_DLL_NAMES)
    for folder in _DLL_DIRS:
        for name in _DLL_NAMES:
            path = Path(folder) / name
            if path.is_file():
                candidates.append(str(path))
    return candidates


def _bind(dll: C.WinDLL):
    for name, argtypes in _SIGNATURES.items():
        func = getattr(dll, name)
        func.argtypes = argtypes
        func.restype = C.c_int


def load_api(dll_path: str | None = None) -> tuple[C.WinDLL, str]:
    """
    Load the first uc480/uEye DLL that reports at least one camera.

    Returns the bound DLL and the path it was loaded from.

    Only one file per DLL name is loaded: a second copy of uc480_64.dll from
    another folder in the same process crashes it (access violation in the
    loader), which is what happens when the first copy sees no camera.
    """
    tried, loaded_names = [], set()
    for path in _candidate_paths(dll_path):
        name = Path(path).name.lower()
        if name in loaded_names:
            tried.append(f"{path}: skipped, {name} is already loaded from another folder")
            continue
        try:
            dll = C.WinDLL(path)
            _bind(dll)
        except OSError as err:
            tried.append(f"{path}: {err.strerror or err}")
            continue
        loaded_names.add(name)

        n_cams = C.c_int(0)
        dll.is_GetNumberOfCameras(C.byref(n_cams))
        if n_cams.value > 0:
            return dll, path
        tried.append(f"{path}: loaded, but sees no camera")

    raise RuntimeError(
        "[CinCam uc480] No uc480 / uEye DLL could see a camera. Tried:\n  "
        + "\n  ".join(tried)
        + "\nIf a DLL loaded but sees no camera: is the CinCam plugged in (Device Manager: "
          "'uc480 cmos, 1280x1024')? If none loaded: install the Thorlabs uc480 driver "
          "(ThorCam) or point the CINCAM_UC480_DLL environment variable at uc480_64.dll."
    )
