# allied-vision-camera

Snapshot-oriented Python wrappers for the yb2 lab's cameras, a live viewer, and PSF plotting helpers.

One canonical camera layer for the yb2 lab — pull it into any project as a git submodule instead of copying camera code around.

**Design:** camera configuration is slow and happens once at construction; snapshots are fast and never re-query hardware (all metadata is cached internally). Every camera has the same `snapshot()` and writes the same folder layout.

## Cameras

| Package | Camera | Driver (install separately) | Docs |
|---|---|---|---|
| `cinogy_uc480` | Cinogy CinCam CMOS 1201 beam profiler, with a live mode | Thorlabs uc480 driver (free ThorCam). No RayCi, no pyueye. | [cinogy_uc480/README.md](cinogy_uc480/README.md) |
| `cinogy` | Cinogy CinCam CMOS 1201 | CINOGY / RayCi install, `pip install -e ".[ueye]"` | [cinogy/README.md](cinogy/README.md) |
| `allied_vision` | Allied Vision Alvium G1, Mako G-131B | Vimba X SDK | this README |
| `camera_live_view` | Live viewer for `cinogy_uc480` (pyqtgraph) | – | [examples/README.md](examples/README.md) |

One example script per camera, and the live viewer, are in [examples/](examples/README.md).

### Quick start: CinCam

1. Install **ThorCam** from Thorlabs (brings the uc480 driver).
2. Clone and install:
   ```bash
   git clone git@github.com:UCBoulder/allied-vision-camera.git
   cd allied-vision-camera
   pip install -r requirements.txt
   ```
3. Close RayCi / ThorCam, then `python examples/live_view.py` or `python examples/snapshot_cincam_uc480.py`.

---

## Prerequisites (Allied Vision)

The rest of this README covers the Allied Vision cameras. The CinCam needs none of it: see
[Quick start: CinCam](#quick-start-cincam).

Python ≥ 3.11, and **the Vimba X SDK installed system-wide**.

`vmbpy` is only a thin Python binding. It talks to the camera through the vendor's
GenTL *transport layer* (`.cti` files), which ships with the SDK, not with pip. Without
the SDK, `import vmbpy` may even succeed while zero cameras are ever found.

### Installing Vimba X

1. Download **Vimba X** for your platform from the
   [Allied Vision software downloads page](https://www.alliedvision.com/en/products/software/vimba-x-sdk/)
   (free, but the site asks for an email). Get Vimba X — the older "Vimba SDK" ships
   the legacy `vimba` Python package, which is *not* `vmbpy` and will not work here.
2. Run the installer. On Windows, include the **USB/GigE driver** component
   (`VimbaXDriverInstaller.exe`) — without it a USB3 camera enumerates as an
   unknown device and never appears.
3. Reboot, or at least open a fresh terminal: the installer sets environment
   variables that existing shells will not have picked up.

The installer sets `VIMBA_X_HOME` and appends its `cti/` directory to
`GENICAM_GENTL64_PATH`. Those two variables are how `vmbpy` finds the hardware.

Default install root:

| | |
|---|---|
| Windows | `C:\Program Files\Allied Vision\Vimba X` |
| Linux | `/opt/VimbaX_<version>` |

### Verifying the install from the terminal

Work down this list — each step isolates a different layer, so the first one that
fails tells you where the problem is.

**1. Is the SDK installed and are its environment variables set?**

```powershell
# Windows PowerShell
$env:VIMBA_X_HOME
$env:GENICAM_GENTL64_PATH
```
```bash
# Linux / macOS
echo $VIMBA_X_HOME
echo $GENICAM_GENTL64_PATH
```

`VIMBA_X_HOME` should print the install root, and `GENICAM_GENTL64_PATH` should
contain that root's `cti` directory. Empty output means the SDK is not installed, or
your shell predates the install — open a new terminal and retry.

**2. Does the SDK itself see the camera?** This is pure C, no Python involved, so it
cleanly separates "SDK/driver problem" from "Python problem":

```powershell
& "$env:VIMBA_X_HOME\bin\ListCameras_VmbC.exe"
```
```bash
$VIMBA_X_HOME/bin/ListCameras_VmbC
```

Expected — note `Transport Layer Path`, which tells you *which* `.cti` found the camera:

```
Vmb Version Major: 1 Minor: 2 Patch: 0
TransportLayers found: 4
Interfaces found: 6
Cameras found: 1

/// Camera Name            : Allied Vision 1800 U-508m (DEV_1AB22C00A1B2)
/// Camera ID              : DEV_1AB22C00A1B2
/// Permitted Access Modes : Full access, Read access, Exclusive access
/// @ Transport Layer Path : C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti
```

`Cameras found: 0` here means the SDK is fine but the camera is not reachable —
cable, power, driver, or firewall (see the table below). Nothing in Python will help.

**3. Does Python see the same thing?**

```bash
python -c "import vmbpy; print(vmbpy.__version__)"
```

Then the end-to-end check — this needs no camera attached and prints both the API
versions and every device found:

```bash
python -c "import vmbpy; vmb = vmbpy.VmbSystem.get_instance()
with vmb:
    print(vmb.get_version())
    for c in vmb.get_all_cameras():
        print(c.get_id(), '|', c.get_model())"
```

Expected:

```
vmbpy: 1.2.1 (using VmbC: 1.3.0, VmbImageTransform: 2.3)
DEV_1AB22C00A1B2 | 1800 U-508m
```

If this prints a version but `import vmbpy` failed at step 3, it is a pip problem;
if it prints a version and no cameras while step 2 found one, it is almost always a
32/64-bit or virtual-environment mismatch.

**4. Sanity-check the GUI.** `VimbaXViewer.exe` (in `bin/`) should show a live image.
Then **close it** — it holds the camera in Full access mode and `AlviumG1` will refuse
to open while it is running. This is the single most common failure in practice.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'vmbpy'` | Pip-level only. `pip install -e .` in the wrong environment, or a venv that is not active. Check `python -c "import sys; print(sys.executable)"`. |
| `VmbSystemError` / `Failed to load VmbC` on import | SDK missing, or installed for the wrong architecture. `vmbpy` needs **64-bit** Python against the 64-bit SDK. Verify with `python -c "import struct; print(struct.calcsize('P')*8)"` → must be `64`. |
| `RuntimeError: No Allied Vision cameras found.` (raised by `AlviumG1`) | Step 2 above is the discriminator. If `ListCameras` also finds nothing: re-seat the USB3 cable (must be a USB3 port — blue, and USB2 cables silently fail), run `VimbaXDriverInstaller.exe`, and for GigE cameras allow Vimba through the firewall and check the NIC is on the camera's subnet. |
| `RuntimeError: Could not open '<id>' in Full access mode.` | Another process holds the camera. Close Vimba X Viewer, and any Python session that constructed an `AlviumG1` without calling `.close()`. A crashed script leaves the camera locked — the lock clears when that process exits. |
| Camera connects but frames are synthetic / the model says **Camera Simulator** | The SDK ships a simulator transport layer (`VimbaCameraSimulatorTL.cti`) that presents fake devices. `AlviumG1` with `camera_id=None` takes whichever camera enumerates **first**, which can be a simulator. Run step 2, find your real camera's ID, and pass it explicitly: `AlviumG1({"camera_id": "DEV_1AB22C00A1B2", ...})`. Worth doing on any machine with more than one camera listed. |
| `VmbTimeout` / `capture_frame()` hangs then raises | The 2000 ms frame timeout is shorter than the exposure. Lower `exposure_us`, or check the camera is not waiting on an external trigger. |
| Other vendors' cameras appear in the list | `GENICAM_GENTL64_PATH` is shared across vendors, so a Thorlabs or IDS install adds its own producers. Harmless, but another reason to pin `camera_id`. |
| Everything worked yesterday, nothing today | Almost always an unclosed handle from a crashed script. Close every Python process using the camera, or unplug/replug it. |

## Installation

As a standalone checkout:

```bash
git clone git@github.com:UCBoulder/allied-vision-camera.git
cd allied-vision-camera
pip install -r requirements.txt     # = pip install -e ".[live]": all cameras + the live viewer
```

For use inside another project, see [Using this repo as a submodule](#using-this-repo-as-a-submodule).

## Usage

```python
from allied_vision import AlviumG1

cam = AlviumG1({
    "data_root": "data",
    "exposure_us": 6000,
    "gain_db": 0.0,
})

img, out_dir = cam.snapshot(
    experiment_name="alignment",
    comment="after realigning L3",
    save_preview=True,
)

cam.close()
```

`img` is a `uint16` NumPy array with values from 0 to `cam.max_value` (e.g. 4095 for Mono12, 1023 for Mono10). `out_dir` is the directory the snapshot was written to.

### Output layout

```
<data_root>/<experiment_name>/snapshot_<YYYY_MM_DD-HH_MM_SS>/
├── data_<stamp|filename>.npy       # raw uint16 frame
├── preview_<stamp|filename>.png    # if save_preview=True
├── preview_3d_<...>.png            # if plot_psf_3d=True
└── metadata_<stamp|filename>.json  # if save_metadata=True (default)
```

## Configuration

Passed as a dict to `AlviumG1(config)`. Only `data_root` is required; every other key falls back to the default in the `_PARAMETERS` registry.

| Key | Default | Meaning |
|---|---|---|
| `data_root` | *(required)* | Root directory for snapshot output. Created if missing. |
| `camera_id` | `None` | Specific camera ID. `None` connects to the first camera found. |
| `exposure_us` | `6000` | Exposure time in µs. Ignored if `exposure_auto` is `True`. |
| `gain_db` | `0.0` | Analog gain in dB. Ignored if `gain_auto` is `True`. |
| `exposure_auto` | `False` | Continuous auto-exposure. |
| `gain_auto` | `False` | Continuous auto-gain. |
| `pixel_format` | `"auto"` | `"auto"` picks the highest Mono bit depth the camera supports (Mono16 → Mono8). Or name one explicitly, e.g. `"Mono10"`; an unsupported format raises `ValueError` listing what the camera offers. |

## API

| | |
|---|---|
| `AlviumG1(config)` | Connect, open in Full access mode, apply settings, cache metadata. |
| `.snapshot(experiment_name, ...)` | Capture and save one frame. Returns `(img, out_dir)`. |
| `.capture_frame()` | Capture one frame in memory, no disk I/O. Returns `uint16` array. |
| `.plot(experiment_name=None, npy_path=None, cmap=None)` | Display a saved snapshot (defaults to the most recent one). |
| `.set_exposure(us)` / `.set_gain(db)` | Change settings at runtime; the metadata cache follows. |
| `.close()` | Reset to Mono8, close the camera, shut down the Vimba system. **Always call this** — an unclosed camera stays locked. |

`snapshot()` also accepts `path` (override the output directory), `filename` (fixed stem instead of a timestamp), `exposure_time_us` (one-shot exposure, restored afterwards), `cropping_params` (`{"offset_px": (x, y), "width_px": w, "height_px": h}`, AOI restored afterwards), `plot_psf_3d`, `save_preview`, `save_metadata`, and `comment`.

### Plotting helpers

Usable standalone on any 2-D array — no camera required:

```python
from allied_vision import save_psf_2d, save_psf_3d

save_psf_2d(img, "preview.png", title="PSF", max_value=cam.max_value)   # beam colours, log scale 0..max_value
save_psf_3d(img, "preview_3d.png", title="PSF")  # 3-D surface
```

## Using this repo as a submodule

This is the intended way to consume the package: each project gets its own checkout,
pinned to a specific commit, so an update here can never silently change the behaviour
of an experiment you ran last month.

### Adding it to a project

From the root of the consuming repo:

```bash
git submodule add git@github.com:UCBoulder/allied-vision-camera.git external/allied_vision_camera
git commit -m "Add allied-vision-camera submodule"
```

That creates two tracked things: a `.gitmodules` file recording the URL and path, and a
special entry at `external/allied_vision_camera` — a *gitlink* storing one commit SHA,
not the files. `external/` is a convention; any path works.

Then make the package importable. Preferred:

```bash
pip install -e external/allied_vision_camera
```

`from allied_vision import AlviumG1` then works from any working directory. The
alternative is adding the submodule path to `PYTHONPATH` (what some older lab projects
do via `.vscode/settings.json`), but it breaks as soon as a script is run from
elsewhere.

### Cloning a project that already has it

A plain `git clone` leaves submodule directories **empty** — this surprises everyone
at least once. Either clone recursively:

```bash
git clone --recurse-submodules git@github.com:UCBoulder/<project>.git
```

or fix up an existing clone:

```bash
git submodule update --init --recursive
```

Requires an SSH key on the collaborator's GitHub account, since `.gitmodules` uses an
SSH URL. Someone cloning over HTTPS without keys gets a `Permission denied (publickey)`
on the submodule step, not on the main clone.

### Pulling in camera updates

A submodule tracks a **commit, not a branch**, so new work here does not reach a
project until someone moves the pointer on purpose:

```bash
git submodule update --remote external/allied_vision_camera   # fast-forward to origin/main
git add external/allied_vision_camera
git commit -m "Bump allied-vision-camera to <short-sha>"
```

The staged change is one line — the old SHA replaced by the new one. Review what you
are pulling in first with `git -C external/allied_vision_camera log --oneline HEAD..origin/main`.

### Making changes to the camera code from inside a project

The submodule is a full git repo, so you can edit and commit inside it — but its `HEAD`
starts **detached**, and committing onto a detached HEAD strands the work where the next
checkout will lose it. Always land on a branch first:

```bash
cd external/allied_vision_camera
git checkout main && git pull        # leave detached HEAD before editing
# ...edit, test...
git commit -am "Fix exposure clamping"
git push origin main                 # push to THIS repo first

cd ../..                             # back to the project
git add external/allied_vision_camera
git commit -m "Bump allied-vision-camera"
```

**Push the submodule before pushing the parent.** If you push only the parent, its
recorded SHA points at a commit nobody else can fetch, and every collaborator's
`submodule update` fails. `git push --recurse-submodules=check` makes git refuse the
parent push when the submodule commit is unpublished — worth setting as a habit.

### Everyday commands

| | |
|---|---|
| `git submodule status` | Show the pinned SHA. A leading `+` means the checkout differs from what the parent records; `-` means it is not initialised. |
| `git diff --submodule` | Show submodule pointer moves as commit lists rather than opaque SHAs. |
| `git submodule update` | Reset the submodule to the SHA the parent records — **discards uncommitted work inside it**. |
| `git submodule foreach git status` | Status across every submodule at once. |
| `git config --global submodule.recurse true` | Make `git pull` update submodules automatically. Recommended; removes the most common source of "it works on my machine". |
| `git config --global diff.submodule log` | Makes the default `git diff` readable for pointer moves. |

### Gotchas

- A dirty submodule shows in the parent as a bare `modified: external/allied_vision_camera (modified content)` with no file detail. Run `git -C external/allied_vision_camera status` to see what actually changed.
- Switching branches in the parent does **not** switch the submodule. After any branch change, run `git submodule update` or you are silently testing against the wrong camera code.
- `git submodule update` is destructive to uncommitted changes inside the submodule. Commit or stash there first.
- Never commit an edit to the camera code only in the parent project's copy — it belongs here, or the next `submodule update` erases it.

## Known constraints

- **Mono (greyscale) formats only.** Plot scaling follows the selected bit depth (`cam.max_value`); `save_psf_2d` / `save_psf_3d` default to 4095 when called directly without `max_value`. `capture_frame()` takes channel 0 of the frame.
- Frame timeout is fixed at 2000 ms.
- Status messages go to `stdout` via `print()`, not the `logging` module.
- Only one camera is opened per `AlviumG1` instance; with no `camera_id` it takes whichever camera enumerates first.

## Provenance

Extracted from `hardware/camera/` in [UCBoulder/DMD](https://github.com/UCBoulder/DMD) as a standalone, reusable package. That repo keeps its own copy for now; this one is the canonical version going forward.
