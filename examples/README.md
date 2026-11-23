# Examples

A snapshot script and a live viewer, both with a `CONFIG` dict at the top; the camera is
picked by name (`cameras.py`, which also holds the Allied Vision settings). Run them from the repo root after `pip install -r requirements.txt`.

## Camera types

| Camera | Package / class | Driver to install (not pip) | Snapshot | Live view |
|---|---|---|---|---|
| **Cinogy CinCam CMOS 1201** beam profiler, via the Thorlabs uc480 driver *(recommended for the CinCam)* | `cinogy_uc480.CinCamUC480` | Thorlabs uc480 driver: install the free ThorCam software (thorlabs.com). No RayCi. | `snapshot.py` | `live_view.py` |
| Cinogy CinCam CMOS 1201, via the IDS uEye API | `cinogy.CinCamCMOS1201` | CINOGY / RayCi install (`ueye_api_64.dll`), plus `pip install -e ".[ueye]"` | `snapshot.py cincam_ueye` | – |
| Allied Vision Alvium G1 / Mako G-131B | `allied_vision.AlviumG1` | [Vimba X SDK](https://www.alliedvision.com/en/products/software/vimba-x-sdk/) (see the main README) | `snapshot.py alvium` | `live_view.py alvium` |
| Thorlabs Zelux CS165MU1 | `thorlabs_zelux.ThorlabsZelux` | ThorImageCAM or ThorCam from thorlabs.com (brings `thorlabs_tsi_camera_sdk.dll`) | `snapshot.py zelux` | `live_view.py zelux` |

The CinCam is an IDS uEye UI-154xLE-M inside (1280 × 1024, 5.2 µm pixels, 8-bit). Both
CinCam packages talk to the same camera through the same uEye C API; they differ only in
which vendor driver owns it. The uc480 one needs nothing but the free Thorlabs driver.

The live viewer is camera-independent: `python examples/live_view.py cincam` (default),
`python examples/live_view.py alvium` for an Allied Vision camera over Vimba X (Alvium,
Mako over GigE), or `python examples/live_view.py zelux`. Any camera class with the small live interface (start_live / stop_live /
grab_latest, exposure, gain, readout region, snapshot) can be plugged in the same way.

## CinCam in three steps

1. Install **ThorCam** from Thorlabs (it brings the uc480 driver). In Device Manager the
   camera then shows as *uc480 cmos, 1280x1024*.
2. `pip install -r requirements.txt` in this repo.
3. `python examples/live_view.py` (or `python examples/snapshot.py`).

The driver DLL is found on its own (ThorCam and Thorlabs Wavefront Sensor install folders,
PATH); otherwise set `CINCAM_UC480_DLL` to the full path of `uc480_64.dll`.

## Zelux

The CS165MU1 is a 1440 × 1080 global-shutter mono sensor (Sony IMX273, 3.45 µm pixels,
10-bit, 0–48 dB gain, exposure 40 µs to 26 s): 35 fps full frame on USB 3, 138 fps on a
320 × 240 readout region. The region snaps to the sensor's steps (minimum 80 × 12 px);
read `cam.aoi` for what was set. The SDK DLL is found in the ThorImageCAM and ThorCam
install folders; otherwise set `THORLABS_TSI_SDK_DLL` to the full path of
`thorlabs_tsi_camera_sdk.dll`. Pick a camera by serial number with `"camera_id": "39882"`
when several are plugged in. A snapshot takes ~0.3 s (the camera is armed for each one).

## Notes

- **One program per camera.** Close RayCi, ThorCam, ThorImageCAM or Vimba X Viewer before
  running a script; a crashed script can also keep the camera until its Python process ends.
- **Mako over GigE:** the packet size must fit the link. The driver lets Vimba negotiate
  it at connection; if frames still arrive incomplete (typical behind a USB-to-Ethernet
  adapter, which drops jumbo packets) set `"packet_size": 1500` in the config. Broken
  frames are never saved: after 20 of them the driver prints their status, the packet size,
  bandwidth and Vimba's packet counters. If packets are lost although they fit the link,
  cap the bandwidth (`"stream_bytes_per_second": 40e6`) or enable jumbo frames on the
  adapter, which cuts the packet rate about six-fold.
- Every camera writes the same folders, `data/<experiment>/snapshot_<timestamp>/` with
  `data_*.npy` (raw frame, uint16), `preview_*.png` and `metadata_*.json`, under
  `DATA_ROOT` set in `examples/cameras.py` (the Objective-testing `data` folder, or this
  repo's `data/` on a PC without it; `data/` is ignored by git).
- `aoi` in `CONFIG` reads out only a rectangle of the sensor (faster, smaller files); the
  rectangle is recorded in the metadata.
- Live view layout: Settings and Alignment are two docks sharing tabs at the top of the
  left column (the tab is the only label while they are tabbed). Drag the separator to
  widen them, drag a tab out to make that dock a window of its own (a dock on its own
  has a title bar again: drag it or double-click it to dock it back), drop it on the
  right, or hide it with its toolbar button to give the image the whole screen. Each
  scrolls when taller than the screen. The window remembers its size, position and dock
  layout between runs; "Reset layout" puts everything back (both tabbed on the left,
  window maximised).
- Live view "Alignment" dock (tab next to Settings; drag it out for its own window): the
  **focal plane scan**. Enter a name, the start position and the step (typed in mm, inch
  or µm; positions are stored and analysed in mm), start the scan, take a snapshot, move
  the camera by the step (or by anything else: "Next snapshot at" is pre-filled with the
  last position + the last step taken and can be edited before each snapshot, so a typed
  custom step is kept for the following suggestions),
  take the next, … and finish. The raw frames go to `data/alignment_data/<name>_<time>/`
  as `data_<position><unit>.npy` next to `scan.json`, which lists every position with the
  exposure, gain and readout region, plus the camera metadata (`camera_live_view.
  position_scan.PositionScan` reads and writes this format). When the scan is finished
  the chosen analysis runs in its own window: `camera_live_view.scan_analysis` fits a
  2D Gaussian to each frame, plots the 1/e² radii along x and y against position, fits a
  parabola to w² (a Gaussian beam's w(z)² is one) and reports its vertex as the focal
  plane, per axis and averaged. `examples/analyze_scan.py` runs the same analysis on a
  saved scan; add your own to `scan_analysis.ANALYSES` (a function `folder -> Figure`)
  and it appears in the dock's menu.
- Live view "Alignment" dock, **PSF vs target and camera position**: place the orange
  analysis square on the PSF ("Centre region on the current fit", or drag it), start, then
  for every shot type the target z and camera z (each with its own unit: mm, inch or µm;
  saved in mm) and take a snapshot; take a background with the PSF light blocked (one per
  exposure and gain used; finishing asks for any missing). Frames go to
  `data/experiment_data/<name>_<time>/` with `scan.json` (both positions, the analysis
  region, exposure, gain and readout region per shot), and a PNG of every frame (the raw
  frame with its colour bar, in the log or linear scale shown on screen, positions and
  exposure in the title) in `PNG/` under the same name. Its analyses are listed apart from
  the focal-plane scan's (`scan_analysis.PSF_POSITION_ANALYSES`) and plugged in by the
  program that starts the viewer: Objective-testing's `scripts/live_view.py` adds NA,
  Strehl ratio and Zernikes vs position (three figures), which runs when the experiment
  is finished. An analysis may return one figure or a list of them. Started from
  `examples/live_view.py` the menu says "none registered".
- Live view "Alignment" dock, **power-meter scan**: where on the camera is the power behind
  an aperture / fibre highest? Needs Thorlabs' Optical Power Monitor software (its TLPM
  driver; close the OPM application itself). "Find", "Connect", choose x or y, start, and
  move the beam slowly across: every Gaussian fit of the spot adds a point (fit centre,
  power averaged over that frame's exposure) to the live plot, with a Gaussian on an
  offset refitted as points come in. "Reset data" drops the points so far, "Reset fit"
  forgets the fit of the axis on the plot and its remembered maximum. "Stop, fit and
  save" fits once more and saves, only then, `<folder>/<name>_<axis>_<time>/` with
  `scan.json` (camera and meter metadata, the fit), `power_scan.csv` (one row per point)
  and `power_vs_<axis>.png`; the folder is chosen in the dock. After an x and a y scan
  the viewer asks whether to mark (x0, y0) as a reference point.
  `camera_live_view.power_scan.load_power_scan(folder)` reads a scan back. Any meter
  object with `power_w()` and `close()` can be passed instead:
  `run_live_view(cam, ..., power_meter=meter)`.
- The highest count in the orange analysis square is written above it in large white
  letters with a black edge, readable on every colour of the scale; magenta with
  "SATURATED" when the raw frame clips in the square.
- Live view mouse: wheel zooms, right-drag pans, left-drag moves or resizes the yellow
  readout rectangle. The save button stores the raw frame even when background
  subtraction is shown, and records the background file in the metadata.
- Live view "Gaussian fit": fits a 2D Gaussian to the spot up to ten times a second and
  shows its centre, 1/e² radii, ellipticity (minor / major radius, ISO 11146) and major-axis angle under the image (pixels or µm on the sensor; origin at
  the top-left pixel or the camera centre) and the ellipse with its principal axes on the image. Useful for
  alignment: watch the centre while you turn a mirror. The fit window can cover the whole
  sensor (±1000 px): large windows are binned so that a fit stays at 10–20 ms, and a
  small spot inside a large window is automatically refitted unbinned. Set the window
  larger than the beam; a window smaller than the beam gives meaningless widths.
  The fit window is a drop-down of presets (±10 to ±500 px around the brightest spot) or
  "full frame" for a beam larger than the sensor; its centre may then lie off the chip and
  is flagged as extrapolated. Any half-size can be typed. Under the current fit the readout
  shows the plain mean and standard deviation of the last N fitted centres (default 10).
  "Reference points" (e.g. the optical axis, for realigning optics): give a name and add the
  point at the current fit centre, at the averaged centre, or at a typed x, y (in the units
  and origin of the fit menu). All points are drawn as coloured crosses with their
  coordinates in the image legend; adding with an existing name moves that point, the
  drop-down removes one, "Remove all" clears them. Points are kept per camera between runs.
- Dead pixels and dark spots (same group): light the whole sensor with a clean, broad
  beam, read out the full sensor and press "Find dead pixels". Eight frames are averaged
  and two kinds of bad pixels are mapped. A *dead pixel* is a single pixel at least 30 %
  darker than every linear prediction from its neighbours, by more than 8 × the measured
  noise, with smooth surroundings (an edge is not a dead pixel). A *dark spot* is a patch
  of 2 or more pixels below 70 % of the smooth local background (a median over 25 px),
  again by more than 8 × the noise: dust or a speck on the sensor window. The map is
  saved as JSON next to the driver (`cinogy_uc480/dead_pixels_<serial>.json`, recorded
  in every snapshot's metadata), loaded again at start, drawn on the image with "Show
  dead pixels" (circles for dead pixels, dots for spot pixels) and, with "Ignore … in
  fits", replaced before the Gaussian fit: dead pixels by the mean of their neighbours,
  spots by the mean of a ring around them. Saved frames are never altered.
