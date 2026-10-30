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
radius is fitted with a parabola w(z) = c (z − a)² + b with c ≥ 0 and b ≥ 0
(bounded least squares); its vertex a is the focus, marked with a vertical line, per
axis and averaged (the two differ for an astigmatic beam). When the vertex
lies outside the scan the plot says so; the error of a comes from the fit
covariance.
"""

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import least_squares

from .dead_pixels import load_dead_pixels, repair
from .gaussian_fit import fit_gaussian_2d
from .position_scan import PositionScan


def parabola_focus(z: np.ndarray, w: np.ndarray) -> dict | None:
    """
    Fit the parabola w(z) = c (z − a)² + b to the radii with c ≥ 0 (opens
    upwards) and b ≥ 0 (least squares with bounds) and return dict(a, a_err,
    b, c, curve(z), inside): the vertex a is the focus, b the radius there;
    `inside` says whether a lies within the scanned range. When the scan only
    shows the radius falling, the vertex lands beyond the last point in that
    direction. None with fewer than 3 points or all at the same position;
    with exactly 3 points the error is undefined (nan).
    """
    z, w = np.asarray(z, float), np.asarray(w, float)
    if len(z) < 3 or np.ptp(z) == 0:
        return None
    # start values: the free parabola if it opens upwards, else a vertex past the
    # end of the scan on the side the radius is falling towards
    p2, p1, p0 = np.polyfit(z, w, 2)
    slope = np.polyfit(z, w, 1)[0]
    if p2 > 0 and p0 - p1 ** 2 / (4 * p2) >= 0:
        start = (-p1 / (2 * p2), p0 - p1 ** 2 / (4 * p2), p2)
    else:
        a0 = z.max() + np.ptp(z) if slope < 0 else z.min() - np.ptp(z)
        start = (a0, 0.5 * w.min(), 0.5 * w.min() / np.ptp(z) ** 2)
    res = least_squares(lambda p: p[2] * (z - p[0]) ** 2 + p[1] - w, x0=start,
                        bounds=((-np.inf, 0.0, 0.0), (np.inf, np.inf, np.inf)), x_scale="jac")
    a, b, c = (float(v) for v in res.x)
    a_err = float("nan")
    dof = len(z) - 3
    if dof > 0:
        try:
            cov = np.linalg.inv(res.jac.T @ res.jac) * (res.fun @ res.fun) / dof
            a_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return dict(a=a, a_err=a_err, b=b, c=c, curve=lambda zz: c * (np.asarray(zz, float) - a) ** 2 + b,
                inside=bool(z.min() <= a <= z.max()))


def fit_scan(folder, position_scale: float = 1.0) -> dict:
    """
    Gaussian fit of every frame of the scan: positions, wx, wy (sensor axes),
    peaks, units. The recorded positions are multiplied by position_scale
    (e.g. 25.4 for a scan whose steps were typed in mm but moved in inches).
    """
    scan = PositionScan.load(folder)
    cam = scan.meta.get("camera", {})
    dead = load_dead_pixels(cam["dead_pixel_file"]) if cam.get("dead_pixel_file") else None
    scale, w_unit = 1.0, "px"          # radii in pixels (pixel size: cam.get("pixel_size_um"))
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
        positions.append(entry["position"] * position_scale)
        wx.append(rx * scale)
        wy.append(ry * scale)
        peaks.append(fit["peak"])
    return dict(folder=str(scan.folder), name=scan.meta.get("name", ""), unit=scan.unit, w_unit=w_unit, px=scale,
                position_scale=position_scale,
                positions=np.array(positions), wx=np.array(wx), wy=np.array(wy), peaks=np.array(peaks),
                failed=failed, max_value=cam.get("max_value"))


def waist_vs_position(folder, fig: Figure | None = None, position_scale: float = 1.0) -> Figure:
    """
    1/e² radii along x and y against position, the fitted caustic of each and
    the focus (its minimum) as a vertical line, plus the mean focal plane.
    Draws into `fig` (e.g. one from pyplot) or a new Figure.
    """
    r = fit_scan(folder, position_scale)
    z, unit = r["positions"], r["unit"]
    fits = {axis: parabola_focus(z, r[f"w{axis}"]) for axis in ("x", "y")}
    foci = [f["a"] for f in fits.values() if f is not None]
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
        err = f" ± {fit['a_err']:.3f}" if np.isfinite(fit["a_err"]) else " (3 pts, no error)"
        where = "" if fit["inside"] else ", outside the scan"
        ax.plot(zz, curves[-1], "-", color=colour, alpha=0.7,
                label=f"fit {axis}: focus a = {fit['a']:+.3f}{err} {unit}{where}; b = {fit['b']:.1f} {r['w_unit']}")
        ax.axvline(fit["a"], color=colour, ls=":", alpha=0.8)
    if len(foci) == 2:
        z_focus = float(np.mean(foci))
        ax.axvline(z_focus, color="k", ls="--", label=f"focal plane {z_focus:+.3f} {unit} (mean of x and y)")
    title = f"{r['name']}: Gaussian waist vs position\nfit per axis: w(z) = c (z − a)² + b, focus at the vertex a"
    if position_scale != 1.0:
        title += f"\nrecorded positions × {position_scale:g}"
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
