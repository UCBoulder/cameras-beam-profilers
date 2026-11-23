"""
Thin ctypes wrapper around Thorlabs' TLPM driver (TLPM_64.dll), only the calls
the live viewer needs. Function names and arguments as in TLPM.h of the
Optical Power Monitor software: ViSession = ViUInt32, ViStatus = ViInt32
(negative = error), ViBoolean = ViUInt16, strings into 1024-byte buffers.

ThorlabsPowerMeter has the power-meter interface of camera_live_view (power_w(),
close(), description, info(), wavelength_nm, set_wavelength_nm()).
"""

import ctypes
from ctypes import byref, c_double, c_int16, c_int32, c_uint16, c_uint32, create_string_buffer
from pathlib import Path

DLL_64 = Path(r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLPM_64.dll")
DLL_32 = Path(r"C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Bin\TLPM_32.dll")
VI_ERROR_RSRC_NFOUND = -1073807343      # 0xBFFF0011: no meter connected
ATTR_SET_VALUE = 0                      # TLPM_ATTR_SET_VAL
POWER_UNIT_WATT = 0                     # TLPM_POWER_UNIT_WATT (1 = dBm)
_BUFFER = 1024

_dll = None


def _tlpm() -> ctypes.CDLL:
    global _dll
    if _dll is None:
        bits64 = ctypes.sizeof(ctypes.c_void_p) == 8
        path = DLL_64 if bits64 else DLL_32
        for candidate in (path, path.name):          # the install folder, then the DLL search path
            try:
                _dll = ctypes.CDLL(str(candidate))
                break
            except OSError:
                continue
        else:
            raise OSError(f"Thorlabs power-meter driver not found ({path}): install Thorlabs' Optical Power "
                          "Monitor software, which installs it.")
    return _dll


def _check(status: int, session: int = 0):
    if status < 0:
        text = create_string_buffer(_BUFFER)
        _tlpm().TLPM_errorMessage(c_uint32(session), c_int32(status), text)
        raise RuntimeError(f"TLPM error {status & 0xFFFFFFFF:#x}: {text.value.decode(errors='replace')}")


def _text(buffer) -> str:
    return buffer.value.decode(errors="replace").strip()


def find_power_meters() -> list[tuple[str, str]]:
    """(resource name, "model serial") of every Thorlabs power meter connected; "(in use)" if another
    program holds it."""
    dll = _tlpm()
    count = c_uint32(0)
    status = dll.TLPM_findRsrc(c_uint32(0), byref(count))
    if status == VI_ERROR_RSRC_NFOUND:
        return []
    _check(status)
    found = []
    for i in range(count.value):
        name, model, serial, maker = (create_string_buffer(_BUFFER) for _ in range(4))
        available = c_uint16(1)
        _check(dll.TLPM_getRsrcName(c_uint32(0), c_uint32(i), name))
        label = _text(name)
        if dll.TLPM_getRsrcInfo(c_uint32(0), c_uint32(i), model, serial, maker, byref(available)) >= 0:
            label = f"{_text(model)} {_text(serial)}".strip() + ("" if available.value else " (in use)")
        found.append((_text(name), label))
    return found


class ThorlabsPowerMeter:
    """One Thorlabs power meter; resource None = the first one find_power_meters() lists. The power unit
    is set to W and auto-ranging on, so readings never clip during a scan."""

    def __init__(self, resource: str | None = None, wavelength_nm: float | None = None):
        dll = _tlpm()
        if resource is None:
            found = find_power_meters()
            if not found:
                raise RuntimeError("No Thorlabs power meter found: is it plugged in and on the TLPM driver?")
            resource = found[0][0]
        self.resource = resource
        self._session = c_uint32(0)
        _check(dll.TLPM_init(resource.encode(), c_uint16(1), c_uint16(0), byref(self._session)))
        try:
            self._setup(wavelength_nm)
        except Exception:
            self.close()                    # else the meter stays held until Python exits
            raise

    def _setup(self, wavelength_nm: float | None):
        self._call("TLPM_setPowerUnit", c_int16(POWER_UNIT_WATT))
        self._call("TLPM_setPowerAutoRange", c_uint16(1))
        if wavelength_nm:
            self.set_wavelength_nm(wavelength_nm)
        maker, device, serial, firmware = (create_string_buffer(_BUFFER) for _ in range(4))
        self._call("TLPM_identificationQuery", maker, device, serial, firmware)
        self.device, self.serial, self.firmware = _text(device), _text(serial), _text(firmware)
        sensor, sensor_serial, message = (create_string_buffer(_BUFFER) for _ in range(3))
        kind, subtype, flags = c_int16(), c_int16(), c_int16()
        try:
            self._call("TLPM_getSensorInfo", sensor, sensor_serial, message, byref(kind), byref(subtype), byref(flags))
            self.sensor = f"{_text(sensor)} {_text(sensor_serial)}".strip()
        except RuntimeError:
            self.sensor = ""
        self.description = f"{self.device} {self.serial}" + (f", sensor {self.sensor}" if self.sensor else "")

    def _call(self, function: str, *args):
        _check(getattr(_tlpm(), function)(self._session, *args), self._session.value)

    def power_w(self) -> float:
        """One reading in W; blocks for the meter's averaging time."""
        power = c_double()
        self._call("TLPM_measPower", byref(power))
        return power.value

    @property
    def wavelength_nm(self) -> float:
        value = c_double()
        self._call("TLPM_getWavelength", c_int16(ATTR_SET_VALUE), byref(value))
        return value.value

    def set_wavelength_nm(self, nm: float):
        """Wavelength of the sensor's responsivity correction."""
        self._call("TLPM_setWavelength", c_double(float(nm)))

    def info(self) -> dict:
        """JSON-safe description, saved with every power-meter scan."""
        return dict(driver="TLPM", resource=self.resource, device=self.device, serial=self.serial,
                    firmware=self.firmware, sensor=self.sensor, wavelength_nm=self.wavelength_nm)

    def close(self):
        if self._session.value:
            _tlpm().TLPM_close(self._session)
            self._session = c_uint32(0)
