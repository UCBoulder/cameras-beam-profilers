# Examples

One snapshot script per camera, all with the same layout (a `CONFIG` dict at the top),
and a live viewer. Run them from the repo root after `pip install -r requirements.txt`.

## Camera types

| Camera | Package / class | Driver to install (not pip) | Snapshot | Live view |
|---|---|---|---|---|
| **Cinogy CinCam CMOS 1201** beam profiler, via the Thorlabs uc480 driver *(recommended for the CinCam)* | `cinogy_uc480.CinCamUC480` | Thorlabs uc480 driver: install the free ThorCam software (thorlabs.com). No RayCi. | `snapshot_cincam_uc480.py` | `live_view.py` |
| Cinogy CinCam CMOS 1201, via the IDS uEye API | `cinogy.CinCamCMOS1201` | CINOGY / RayCi install (`ueye_api_64.dll`), plus `pip install -e ".[ueye]"` | `snapshot_cincam.py` | – |
| Allied Vision Alvium G1 / Mako G-131B | `allied_vision.AlviumG1` | [Vimba X SDK](https://www.alliedvision.com/en/products/software/vimba-x-sdk/) (see the main README) | `snapshot_alvium.py` | – |

The CinCam is an IDS uEye UI-154xLE-M inside (1280 × 1024, 5.2 µm pixels, 8-bit). Both
CinCam packages talk to the same camera through the same uEye C API; they differ only in
which vendor driver owns it. The uc480 one needs nothing but the free Thorlabs driver and
is the only one with a live mode.

## CinCam in three steps

1. Install **ThorCam** from Thorlabs (it brings the uc480 driver). In Device Manager the
   camera then shows as *uc480 cmos, 1280x1024*.
2. `pip install -r requirements.txt` in this repo.
3. `python examples/live_view.py` (or `python examples/snapshot_cincam_uc480.py`).

The driver DLL is found on its own (ThorCam and Thorlabs Wavefront Sensor install folders,
PATH); otherwise set `CINCAM_UC480_DLL` to the full path of `uc480_64.dll`.

## Notes

- **One program per camera.** Close RayCi, ThorCam or Vimba X Viewer before running a
  script; a crashed script can also keep the camera until its Python process ends.
- Every camera writes the same folders, `data/<experiment>/snapshot_<timestamp>/` with
  `data_*.npy` (raw frame, uint16), `preview_*.png` and `metadata_*.json`. `data/` is
  ignored by git.
- `aoi` in `CONFIG` reads out only a rectangle of the sensor (faster, smaller files); the
  rectangle is recorded in the metadata.
- Live view mouse: wheel zooms, right-drag pans, left-drag moves or resizes the yellow
  readout rectangle. The save button stores the raw frame even when background
  subtraction is shown, and records the background file in the metadata.
- Live view "Gaussian fit": fits a 2D Gaussian to the spot up to ten times a second and
  shows its centre and 1/e² radii under the image (pixels or µm on the sensor; origin at
  the top-left pixel or the camera centre) and the ellipse on the image. Useful for
  alignment: watch the centre while you turn a mirror. The fit window can cover the whole
  sensor (±1000 px): large windows are binned so that a fit stays at 10–20 ms, and a
  small spot inside a large window is automatically refitted unbinned. Set the window
  larger than the beam; a window smaller than the beam gives meaningless widths.
