"""
Thorlabs power meters (PM100D / PM100A / PM100USB, PM101-103, PM16 / PM160,
PM400, ...) through Thorlabs' TLPM driver.

    from thorlabs_power_meter import ThorlabsPowerMeter, find_power_meters
    find_power_meters()            # [("USB0::0x1313::0x8078::P0012345::INSTR", "PM100D P0012345"), ...]
    pm = ThorlabsPowerMeter()      # the first one found, or ThorlabsPowerMeter(resource)
    pm.set_wavelength_nm(520)
    pm.power_w()                   # one reading in W (takes the meter's averaging time)
    pm.close()

Needs Thorlabs' Optical Power Monitor (OPM) software, which installs the
driver (TLPM_64.dll in C:/Program Files/IVI Foundation/VISA/Win64/Bin).
Close the OPM application before connecting: only one program can hold the
meter. A meter that is not found may be on the other USB driver: switch it to
TLPM with Thorlabs' "Power Meter Driver Switcher".
"""

from .tlpm import ThorlabsPowerMeter, find_power_meters

__all__ = ["ThorlabsPowerMeter", "find_power_meters"]
