# cinogy-beam-profiler

Snapshot-oriented wrapper for the **Cinogy CinCam CMOS 1201** beam profiler, with the
same interface as `allied_vision.AlviumG1`.

The CinCam CMOS 1201 is an IDS uEye **UI-154xLE-M** inside (1280 x 1024 mono, 5.2 µm
pixels), so it is driven straight through the IDS uEye API via `pyueye`. RayCi itself is
not used.

## Requirements

- The CINOGY / RayCi install, which provides the uEye USB driver and
  `C:\Windows\System32\ueye_api_64.dll`.
- **RayCi must be closed.** Only one program can hold the camera.

## Usage

```python
from cinogy import CinCamCMOS1201

cam = CinCamCMOS1201({"data_root": "data", "exposure_us": 6000, "gain_db": 0.0})
img, out_dir = cam.snapshot(experiment_name="beam_profile", save_preview=True)
cam.close()
```

## Notes

- **8-bit only.** The sensor rejects every Mono10/12/16 mode, so frames are uint16 arrays
  holding 0-255 and `max_value == 255`.
- Exposure is limited by the frame time, so `set_exposure` lowers the frame rate first.
  The exposure the camera actually set (rounded to its line time) is what goes into the
  metadata.
- `gain_db` maps to the uEye master gain factor (100 = 1x).
- `save_preview=True` uses the PSF plot helpers from the `allied_vision` package.
