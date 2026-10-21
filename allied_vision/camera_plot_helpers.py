"""
Camera visualization helpers for PSF analysis.

save_psf_2d  — 2-D plot: beam-profiler colours on a logarithmic scale fixed
    to 0–max_value (white = saturated), the same scale as the live view.

save_psf_3d  — 3-D surface plot, Blues fixed to 0–max_value.

max_value is the sensor's full scale, e.g. 4095 for Mono12, 1023 for Mono10.

Usage
-----
from .camera_plot_helpers import save_psf_2d, save_psf_3d

save_psf_2d(img, Path("output/preview.png"), title="my experiment")
save_psf_3d(img, Path("output/preview_3d.png"), title="my experiment")
"""

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers the 3D projection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UINT12_MAX: int = 4095


def _fixed_scale_ticks(max_value: int) -> list[int]:
    return [int(round(v)) for v in np.linspace(0, max_value, 5)]


def _cbar_label(max_value: int) -> str:
    return f"Intensity (counts, {int(max_value).bit_length()}-bit)"

# Beam-profiler colours, the same as the live view and PSFPlotter.BEAM_CMAP:
# dark blue → blue → cyan → green → yellow → red, white at saturation.
BEAM_CMAP = mcolors.LinearSegmentedColormap.from_list("beam", [
    (0.0, "#00004d"), (0.15, "#0000ff"), (0.35, "#00ffff"), (0.5, "#00ff00"),
    (0.65, "#ffff00"), (0.85, "#ff0000"), (1.0, "#ffffff"),
])

# Default colormap for AlviumG1.plot(): black → blue → green.
# Dark background keeps faint PSF wings visible; used with a fixed full-scale range.
PSF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "psf", ["black", "#0b3d91", "#00a0ff", "#00ff88"]
)


# ---------------------------------------------------------------------------
# 2-D plot
# ---------------------------------------------------------------------------

def save_psf_2d(
    img: np.ndarray,
    save_path: Path,
    title: str = "",
    max_value: int = UINT12_MAX,
) -> None:
    """
    Save the frame as a PNG: one panel in the beam-profiler colours on a
    logarithmic scale fixed to 0–max_value (white = saturated). Every
    snapshot shares the same scale, and faint wings stay visible.

    Parameters
    ----------
    img : np.ndarray
        Raw camera image (uint16).
    save_path : Path
        Destination PNG path.
    title : str
        Optional title; the peak value (and a saturation warning) is added below it.
    max_value : int
        Full-scale value (default 4095, i.e. Mono12).
    """
    img_max = int(img.max())
    h_px, w_px = img.shape
    panel_h = 5.0                                              # inches
    fig_w = min(max(w_px / h_px * panel_h, 3.0) + 1.5, 16.0)  # + colorbar, capped

    fig, ax = plt.subplots(figsize=(fig_w, panel_h + 0.8), constrained_layout=True)
    im = ax.imshow(
        img, cmap=BEAM_CMAP, norm=mcolors.FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max_value),
        origin="upper", interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_ticks([t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if t < max_value]
                   + [max_value])
    cbar.set_label(_cbar_label(max_value) + ", log scale", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("x (pixels)", fontsize=8)
    ax.set_ylabel("y (pixels)", fontsize=8)
    ax.tick_params(labelsize=7)

    peak = f"peak {img_max} / {max_value}" + ("   SATURATED" if img_max >= 0.99 * max_value else "")
    ax.set_title(f"{title}\n{peak}" if title else peak, fontsize=9)

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3-D surface plot
# ---------------------------------------------------------------------------

def save_psf_3d(
    img: np.ndarray,
    save_path: Path,
    title: str = "",
    downsample: int = 4,
    z_percentile_low: float = 1.0,
    z_percentile_high: float = 99.9,
    max_value: int = UINT12_MAX,
) -> None:
    """
    Save a 3-D surface PSF plot as PNG.

    The surface is coloured with the same fixed PSF colormap (0–max_value).
    The Z axis is trimmed to the data's own dynamic range so the relief is
    clearly visible even for weak PSFs.

    Parameters
    ----------
    img : np.ndarray
        Raw camera image (uint16).
    save_path : Path
        Destination PNG path.
    title : str
        Optional plot title.
    downsample : int
        Spatial downsampling factor applied before surface rendering (default 4).
        Reduces vertex count from ~5 M to ~300 k for a typical 2448×2048 sensor.
    z_percentile_low : float
        Lower percentile used to set the Z axis floor (default 1.0).
    z_percentile_high : float
        Upper percentile used to set the Z axis ceiling (default 99.9).
        Keeps the relief visible without being dominated by hot pixels.
    max_value : int
        Full-scale value for the colour normalisation (default 4095, i.e. Mono12).
    """
    ds = max(1, int(downsample))
    sub = img[::ds, ::ds].astype(np.float32)
    h, w = sub.shape

    # Pixel coordinate grids (in original pixel units)
    x = np.arange(w) * ds
    y = np.arange(h) * ds
    X, Y = np.meshgrid(x, y)

    # Face colours: fixed 0–max_value normalisation
    norm = mcolors.Normalize(vmin=0, vmax=max_value)
    face_colors = plt.cm.Blues(norm(sub))

    # Z axis: show only the relevant relief portion
    z_lo = float(np.percentile(sub, z_percentile_low))
    z_hi = float(np.percentile(sub, z_percentile_high))
    z_lo = max(z_lo, 0.0)
    z_hi = max(z_hi, z_lo + 1.0)   # guard against degenerate (flat) images

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, sub,
        facecolors=face_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_zlabel("Intensity (counts)")

    # Colorbar through a ScalarMappable (required for surface with facecolors)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(_cbar_label(max_value), fontsize=9)
    cbar.set_ticks(_fixed_scale_ticks(max_value))

    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
