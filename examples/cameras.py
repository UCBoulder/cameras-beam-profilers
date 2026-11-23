"""
Open a camera by name; shared by the examples (snapshot.py, live_view.py).

    cincam       Cinogy CinCam CMOS 1201 through the Thorlabs uc480 driver (recommended)
    cincam_ueye  the same CinCam through the CINOGY / RayCi uEye driver (pip install -e ".[ueye]")
    alvium       Allied Vision (Alvium G1, Mako over GigE) through Vimba X
    zelux        Thorlabs Zelux (CS165MU1) through the Thorlabs camera SDK of ThorImageCAM / ThorCam

Each driver is imported only when its camera is opened, so a PC needs only
the vendor driver of the camera it uses.
"""

from pathlib import Path

# Where snapshots, scans and experiments are saved: the Objective-testing data folder,
# or this repo's data/ on a PC without it
DATA_ROOT = Path(r"C:\Users\kaiga\Documents\User\PhD\Coding-projects\Objective-testing\data")
if not DATA_ROOT.is_dir():
    DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Extra settings for Allied Vision cameras
ALVIUM = {
    "camera_id": None,       # None: the first real camera found (Vimba X simulators are skipped)
    "pixel_size_um": None,   # e.g. 3.45 for a Mako G-319B, 5.3 for a G-131B; Vimba does not report it
    "use_camera_roi": False, # start with the full sensor
    "pixel_format": "auto",  # "auto" = highest Mono bit depth, or e.g. "Mono8" (half the data of Mono10/12)
    "packet_size": None,     # GigE (Mako): None lets Vimba pick the packet size the link carries;
                             # 1500 if frames still arrive incomplete (USB-to-Ethernet adapters)
    "stream_bytes_per_second": None,  # GigE bandwidth cap; None: 40e6 with standard 1500-byte packets (no
                                      # jumbo frames), the camera's setting otherwise. Lower it (e.g. 25e6) if
                                      # frames still arrive Incomplete
}

CAMERAS = ("cincam", "cincam_ueye", "alvium", "zelux")


def open_camera(kind: str, data_root, exposure_us: float, gain_db: float):
    config = {"data_root": data_root, "exposure_us": exposure_us, "gain_db": gain_db}
    if kind == "cincam":
        from cinogy_uc480 import CinCamUC480
        return CinCamUC480(config)
    if kind == "cincam_ueye":
        from cinogy import CinCamCMOS1201
        return CinCamCMOS1201(config)
    if kind == "alvium":
        from allied_vision import AlviumG1
        return AlviumG1({**config, **ALVIUM})
    if kind == "zelux":
        from thorlabs_zelux import ThorlabsZelux
        return ThorlabsZelux(config)
    raise SystemExit(f"Unknown camera '{kind}': use one of {', '.join(CAMERAS)}")
