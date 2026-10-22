# cinogy-beam-profiler-uc480

Snapshot-oriented wrapper for the **Cinogy CinCam CMOS 1201** beam profiler that does
**not** need RayCi, the CINOGY driver install, or `pyueye`. Same interface as
`cinogy.CinCamCMOS1201` and `allied_vision.AlviumG1`.

This is a separate package from `../cinogy_beam_profiler`. Neither one changes the other,
so use whichever matches the driver installed on the PC.

## Why not the Windows webcam driver?

The CinCam is an IDS uEye UI-154xLE-M inside (USB `VID_1409 PID_1545`). It is a
**vendor-specific USB device (class `FF`), not a UVC webcam**, so DirectShow,
Media Foundation and OpenCV cannot see it. What *does* work without the
proprietary software is the **Thorlabs uc480 driver**. Thorlabs rebrands IDS uEye
cameras (DCx, wavefront sensors, ThorCam), and `uc480_64.dll` exposes exactly the
same C API as IDS's `ueye_api_64.dll`. This package binds it with `ctypes`.

## Requirements

- The Thorlabs uc480 driver, from either
  - ThorCam (free, thorlabs.com), or
  - the Thorlabs Wavefront Sensor software (this is what installed it on the lab PC).
- In Device Manager the camera should show as **"uc480 cmos, 1280x1024, ers"** with
  driver service `uc480`.
- Only one program can hold the camera, so close RayCi / ThorCam first.

The DLL is found automatically, in this order:
1. `config["dll_path"]`
2. the `CINCAM_UC480_DLL` environment variable
3. `uc480_64.dll` / `ueye_api_64.dll` on PATH / System32
4. the ThorCam and Wavefront Sensor install folders

It keeps the first DLL that actually sees a camera. `ueye_api_64.dll` is also tried, so
this package also works on a PC where the IDS/CINOGY driver owns the camera.

## Usage

```python
from cinogy_uc480 import CinCamUC480

cam = CinCamUC480({
    "data_root": "data",
    "exposure_us": 6000,
    "gain_db": 0.0,
    "black_level": 10,       # optional: fixed offset, turns off auto black level
    "pixel_clock_mhz": None, # optional: lower clock -> longer max exposure
})
img, out_dir = cam.snapshot(experiment_name="beam_profile", save_preview=True)
cam.close()
```

## Live mode

Used by the live viewer (`camera_live_view`, see `examples/live_view.py`). The camera
runs free into a ring of
driver buffers and queues each finished frame; `grab_latest()` returns the newest
frame or `None`, never blocking, and skips frames nobody collected:

```python
cam.start_live()
while viewing:
    img = cam.grab_latest()      # None if no new frame since the last call
cam.stop_live()
```

- `set_aoi(offset_px, width_px, height_px)` / `reset_aoi()` change the readout
  rectangle (clamped and aligned to the sensor increments; read `cam.aoi` for the
  result). They pause and restart a running stream (~85 ms) and re-apply the
  exposure, which the frame-time range of the new AOI may otherwise clamp.
- `exposure_range_us()`, `gain_range_db()`, `exposure_us`, `gain_db`, `frame_rate()`.
- `snapshot()` / `capture_frame()` work while live: the stream is paused for the
  capture so the frame carries the settings in force at the call.
- Measured: 14 fps full frame, 155 fps on a 300 x 200 AOI; a setting changed during
  live shows two frames later.

## Notes

- **8-bit only.** The sensor rejects Mono10/12/16 (error 174), so frames are uint16
  arrays holding 0-255 and `max_value == 255`.
- **New settings take effect one frame late** (rolling shutter). After any setting
  change the first frame is discarded automatically.
- The camera powers up at the maximum exposure (~100 ms), so the very first frame
  after opening is usually saturated. The stale-frame discard handles this.
- **Auto black level is on by default.** For quantitative beam profiles, set
  `black_level` so the dark offset does not drift between snapshots.
- Exposure is limited by the frame time, so `set_exposure` lowers the frame rate first.
  At the default 24 MHz pixel clock the frame time is 72-205 ms. Lower
  `pixel_clock_mhz` for longer exposures. The exposure actually set (rounded to the
  line time, 66.75 us) is what goes into the metadata.
- `gain_db` maps to the uEye master gain factor (100 = 1x).

## Dead pixels

`dead_pixels_<serial>.json` in this folder is the camera's dead-pixel map, written by
the live viewer ("Find dead pixels", see `examples/README.md`) and read by it at start;
`cam.dead_pixel_file` gives the path and every snapshot's metadata records it under
`dead_pixel_file`. Frames are never altered: the map is for analysis code
(`camera_live_view.dead_pixels.load_dead_pixels` / `repair`).
