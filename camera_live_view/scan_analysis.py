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
radius follows w(z)² = w0² + θ²(z − z0)²: an upward parabola in z with the
offset w0². That parabola (w0, z0, θ ≥ 0) is fitted to the radii by least
squares; its minimum z0 is the focus, marked with a vertical line, per axis
and averaged (the two differ for an astigmatic beam). When the scan does
not reach the focus the fit extrapolates and says so; the error of z0
comes from the fit covariance.
"""

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import least_squares

from .dead_pixels import load_dead_pixels, repair
from .gaussian_fit import fit_gaussian_2d
from .position_scan import PositionScan


def caustic(z, w0, z0, theta):
    """w(z) = sqrt(w0² + θ² (z − z0)²): the upward parabola in w² with offset w0²."""
    return np.sqrt(w0 ** 2 + theta ** 2 * (np.asarray(z, float) - z0) ** 2)


def parabola_focus(z: np.ndarray, w: np.ndarray) -> dict | None:
    """
    Least-squares fit of caustic() to the radii w(z) with w0 ≥ 0 and θ ≥ 0,
    so the parabola in w² always opens upwards. Returns dict(z0, z0_err, w0,
    w0_err, theta, curve(z), inside): the minimum z0 is the focus; `inside`
    says whether it lies within the scanned range. None with fewer than 3
    points (3 parameters) or all at the same position; with exactly 3
    points the errors are undefined (nan).
    """
    z, w = np.asarray(z, float), np.asarray(w, float)
    if len(z) < 3 or np.ptp(z) == 0:
        return None
    # start values: the vertex of the free quadratic in w² when it opens upwards,
    # else the zero of the straight line through w(z) (the beam converges towards it)
    a, b, c = np.polyfit(z, w ** 2, 2)
    slope, intercept = np.polyfit(z, w, 1)
    if a > 0:
        z0_start, w0_start = -b / (2 * a), np.sqrt(max(c - b ** 2 / (4 * a), 0.0))
    else:
        z0_start, w0_start = (-intercept / slope if slope != 0 else z.mean()), 0.2 * w.min()
    theta_start = max(abs(slope), np.ptp(w) / np.ptp(z), 1e-6 * w.max() / np.ptp(z))
    res = least_squares(lambda p: caustic(z, *p) - w, x0=(max(w0_start, 1e-3 * w.min()), z0_start, theta_start),
                        bounds=((0.0, -np.inf, 0.0), (np.inf, np.inf, np.inf)), x_scale="jac")
    w0, z0, theta = (float(v) for v in res.x)
    errs = (float("nan"), float("nan"))
    dof = len(z) - 3
    if dof > 0:
        try:
            cov = np.linalg.inv(res.jac.T @ res.jac) * (res.fun @ res.fun) / dof
            errs = (float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])))
        except np.linalg.LinAlgError:
            pass
    return dict(z0=z0, z0_err=errs[1], w0=w0, w0_err=errs[0], theta=theta,
                curve=lambda zz: caustic(zz, w0, z0, theta), inside=bool(z.min() <= z0 <= z.max()))


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
    return dict(folder=str(scan.folder), name=scan.meta.get("name", ""), unit=scan.unit, w_unit=w_unit, px=scale,
                positions=np.array(positions), wx=np.array(wx), wy=np.array(wy), peaks=np.array(peaks),
                failed=failed, max_value=cam.get("max_value"))


def waist_vs_position(folder, fig: Figure | None = None) -> Figure:
    """
    1/e² radii along x and y against position, the fitted caustic of each and
    the focus (its minimum) as a vertical line, plus the mean focal plane.
    Draws into `fig` (e.g. one from pyplot) or a new Figure.
    """
    r = fit_scan(folder)
    z, unit = r["positions"], r["unit"]
    fits = {axis: parabola_focus(z, r[f"w{axis}"]) for axis in ("x", "y")}
    foci = [f["z0"] for f in fits.values() if f is not None]
    # x window: the scanned positions and the foci (extrapolated or not), plus 5 %
    x_lo, x_hi = min([z.min()] + foci), max([z.max()] + foci)
    pad = 0.05 * max(x_hi - x_lo, 1e-9)
    x_lo, x_hi = x_lo - pad, x_hi + pad
    fig = fig if fig is not None else Figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    curves = []
    for axis, colour in (("x", "C0"), ("y", "C3")):
        w = r[f"w{axis}"]
        ax.plot(z, w, "o", color=colour, label=f"w{axis} (1/e² radius along {axis})")
        fit = fits[axis]
        if fit is None:
            ax.plot([], [], " ", label=f"w{axis}: no fit (needs 3 positions)")
            continue
        zz = np.linspace(x_lo, x_hi, 400)
        curves.append(fit["curve"](zz))
        err = f" ± {fit['z0_err']:.3f}" if np.isfinite(fit["z0_err"]) else " (3 points: no error estimate)"
        where = "" if fit["inside"] else "  — outside the scanned range, extrapolated"
        ax.plot(zz, curves[-1], "-", color=colour, alpha=0.7,
                label=f"fit w{axis}² = w0² + θ²(z − z0)²: focus z0 = {fit['z0']:+.3f}{err} {unit}, "
                      f"w0 = {fit['w0']:.1f} {r['w_unit']}{where}")
        ax.axvline(fit["z0"], color=colour, ls=":", alpha=0.8)
    if len(foci) == 2:
        z_focus = float(np.mean(foci))
        ax.axvline(z_focus, color="k", ls="--", label=f"estimated focal plane {z_focus:+.3f} {unit} (mean of x and y)")
    title = f"{r['name']}: Gaussian waist vs position"
    if r["failed"]:
        title += f"   (no fit at {', '.join(f'{p:+.3f}' for p in r['failed'])} {unit})"
    if r["max_value"] and len(r["peaks"]) and r["peaks"].max() >= 0.99 * r["max_value"]:
        title += "   SATURATED frames"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"position ({unit})")
    ax.set_ylabel(f"1/e² radius ({r['w_unit']})")
    # initial ranges: a few pixels around the measured radii and the fitted curves; positions and foci along x
    margin = 5 * r["px"]
    w_all = np.concatenate([r["wx"], r["wy"]] + curves)
    if len(w_all):
        ax.set_ylim(max(w_all.min() - margin, 0), w_all.max() + margin)
    if len(z) > 1:
        ax.set_xlim(x_lo, x_hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


ANALYSES = {
    "Gaussian waist vs position (parabola focus)": waist_vs_position,
}
