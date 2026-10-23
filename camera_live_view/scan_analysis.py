"""
Analyses of a position scan (position_scan.py), each a function
`analysis(folder) -> matplotlib Figure`, listed in ANALYSES so the live
viewer can run one when a scan is finished and a script can run the same
one later:

    from camera_live_view.scan_analysis import ANALYSES, waist_vs_position
    fig = waist_vs_position("data/alignment_data/focus_scan_2026_10_22-10_15_00")

waist_vs_position: a 2D Gaussian is fitted to every frame (gaussian_fit.py,
dead pixels repaired when the scan recorded a map) and the 1/e² radii along
the sensor x and y axes are plotted against position. A Gaussian beam's
radius follows w(z)² = w0² + θ²(z − z0)², a parabola in z, so a parabola
is fitted to w² (linear least squares, no start values) and its vertex is
the focus: z0 = −b / 2a with w² = a z² + b z + c, error from the fit
covariance. The focus is given per axis and averaged; the two differ for an
astigmatic beam.
"""

import numpy as np
from matplotlib.figure import Figure

from .dead_pixels import load_dead_pixels, repair
from .gaussian_fit import fit_gaussian_2d
from .position_scan import PositionScan


def parabola_focus(z: np.ndarray, w: np.ndarray) -> dict | None:
    """
    Fit w² = a z² + b z + c and return the vertex: dict(z0, z0_err, w0, a, b, c,
    curve(z)). None with fewer than 3 points or when the parabola opens downwards.
    """
    z, w = np.asarray(z, float), np.asarray(w, float)
    if len(z) < 3 or np.ptp(z) == 0:
        return None
    (a, b, c), cov = np.polyfit(z, w ** 2, 2, cov=len(z) > 3) if len(z) > 3 else (np.polyfit(z, w ** 2, 2), None)
    if a <= 0:
        return None
    z0 = -b / (2 * a)
    w0_sq = c - b ** 2 / (4 * a)
    z0_err = float("nan")
    if cov is not None:
        grad = np.array([b / (2 * a ** 2), -1 / (2 * a), 0.0])     # d z0 / d(a, b, c)
        z0_err = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    return dict(z0=float(z0), z0_err=z0_err, w0=float(np.sqrt(max(w0_sq, 0.0))), a=float(a), b=float(b), c=float(c),
                curve=lambda zz: np.sqrt(np.maximum(a * np.asarray(zz) ** 2 + b * np.asarray(zz) + c, 0.0)))


def fit_scan(folder) -> dict:
    """Gaussian fit of every frame of the scan: positions, wx, wy (sensor axes), peaks, units."""
    scan = PositionScan.load(folder)
    cam = scan.meta.get("camera", {})
    dead = load_dead_pixels(cam["dead_pixel_file"]) if cam.get("dead_pixel_file") else None
    pixel_um = cam.get("pixel_size_um")
    scale, w_unit = (float(pixel_um), "µm") if pixel_um else (1.0, "px")
    positions, wx, wy, peaks, failed = [], [], [], [], []
    for entry in scan.entries:
        frame = scan.frame(entry)
        if dead is not None:
            aoi = entry.get("aoi") or {}
            frame = repair(frame, dead, offset=(aoi.get("offset_y", 0), aoi.get("offset_x", 0)))
        try:
            fit = fit_gaussian_2d(frame, half_size=max(frame.shape))
        except RuntimeError:
            failed.append(entry["position"])
            continue
        # radii along the sensor axes: the fitted axis closest to x is "x"
        rx, ry = (fit["wx"], fit["wy"]) if abs(fit["theta"]) <= np.pi / 4 else (fit["wy"], fit["wx"])
        positions.append(entry["position"])
        wx.append(rx * scale)
        wy.append(ry * scale)
        peaks.append(fit["peak"])
    return dict(folder=str(scan.folder), name=scan.meta.get("name", ""), unit=scan.unit, w_unit=w_unit,
                positions=np.array(positions), wx=np.array(wx), wy=np.array(wy), peaks=np.array(peaks),
                failed=failed, max_value=cam.get("max_value"))


def waist_vs_position(folder, fig: Figure | None = None) -> Figure:
    """
    1/e² radii along x and y against position, parabola fits of w² and the
    estimated focus. Draws into `fig` (e.g. one from pyplot) or a new Figure.
    """
    r = fit_scan(folder)
    z, unit = r["positions"], r["unit"]
    fits = {axis: parabola_focus(z, r[f"w{axis}"]) for axis in ("x", "y")}
    fig = fig if fig is not None else Figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    lines = []
    for axis, colour in (("x", "C0"), ("y", "C3")):
        w = r[f"w{axis}"]
        ax.plot(z, w, "o", color=colour, label=f"w{axis} (1/e² radius along {axis})")
        fit = fits[axis]
        if fit is not None:
            zz = np.linspace(z.min() - 0.1 * np.ptp(z), z.max() + 0.1 * np.ptp(z), 300)
            err = f" ± {fit['z0_err']:.3f}" if np.isfinite(fit["z0_err"]) else " (3 points: no error estimate)"
            outside = not (z.min() - np.ptp(z) <= fit["z0"] <= z.max() + np.ptp(z))
            ax.plot(zz, fit["curve"](zz), "-", color=colour, alpha=0.7,
                    label=f"parabola fit of w{axis}²: focus at {fit['z0']:+.3f}{err} {unit}, "
                          f"w0 = {fit['w0']:.1f} {r['w_unit']}" + ("  — far outside the scan, extend it" if outside else ""))
            ax.axvline(fit["z0"], color=colour, ls=":", alpha=0.7)
            lines.append(fit["z0"])
    if len(lines) == 2:
        z_focus = float(np.mean(lines))
        ax.axvline(z_focus, color="k", ls="--", label=f"estimated focal plane {z_focus:+.3f} {unit} (mean of x and y)")
    title = f"{r['name']}: Gaussian waist vs position"
    if r["failed"]:
        title += f"   (no fit at {', '.join(f'{p:+.3f}' for p in r['failed'])} {unit})"
    if r["max_value"] and len(r["peaks"]) and r["peaks"].max() >= 0.99 * r["max_value"]:
        title += "   SATURATED frames"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"position ({unit})")
    ax.set_ylabel(f"1/e² radius ({r['w_unit']})")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


ANALYSES = {
    "Gaussian waist vs position (parabola focus)": waist_vs_position,
}
