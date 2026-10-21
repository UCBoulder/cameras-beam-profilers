"""
2D Gaussian fit of the brightest spot in a frame, for the live viewer.

    fit = fit_gaussian_2d(frame, half_size=30)
    fit["x0"], fit["y0"]      centre in array coordinates (pixel i centred at i)
    fit["wx"], fit["wy"]      1/e² radii (px) along the ellipse axes, fit["theta"] their rotation

Cheap enough for a few fits per second: a (2 half_size + 1)² window around
the brightest pixel (after a 3 × 3 box blur, so a hot pixel cannot win),
start values from the moments above half maximum, bounded least squares.
"""

import numpy as np
from scipy.optimize import curve_fit


def gaussian_2d(coords, A, x0, y0, wx, wy, theta, B):
    """Rotated 2D Gaussian with 1/e² radii wx, wy (beam-optics convention) on a constant B."""
    x, y = coords
    dx, dy = x - x0, y - y0
    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)
    return A * np.exp(-2.0 * (xp ** 2 / wx ** 2 + yp ** 2 / wy ** 2)) + B


def fit_gaussian_2d(frame: np.ndarray, half_size: int = 30, max_nfev: int = 60) -> dict:
    """
    Fit gaussian_2d() to the spot around the brightest pixel of `frame`.

    Returns a dict with x0, y0, wx, wy, theta, amplitude, offset, peak (A + B)
    and window (x_min, y_min, x_max, y_max); raises RuntimeError if the fit
    does not converge.
    """
    img = np.asarray(frame, dtype=float)
    h_img, w_img = img.shape
    blurred = img[:-2, :-2] + img[:-2, 1:-1] + img[:-2, 2:] + img[1:-1, :-2] + img[1:-1, 1:-1] + img[1:-1, 2:] \
        + img[2:, :-2] + img[2:, 1:-1] + img[2:, 2:]
    iy, ix = np.unravel_index(int(np.argmax(blurred)), blurred.shape)
    iy, ix = iy + 1, ix + 1
    x_min, x_max = max(ix - half_size, 0), min(ix + half_size + 1, w_img)
    y_min, y_max = max(iy - half_size, 0), min(iy + half_size + 1, h_img)
    window = img[y_min:y_max, x_min:x_max]
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    # start values: offset from the low percentile, centre and widths from the moments above half maximum
    offset0 = float(np.percentile(window, 10))
    amplitude0 = float(window.max() - offset0)
    weights = np.clip(window - offset0 - 0.5 * amplitude0, 0, None)
    total = weights.sum()
    if amplitude0 <= 0 or total <= 0:
        raise RuntimeError("no spot in the frame")
    cx, cy = float((weights * xx).sum() / total), float((weights * yy).sum() / total)
    w0x = max(2.0 * np.sqrt((weights * (xx - cx) ** 2).sum() / total), 1.0)
    w0y = max(2.0 * np.sqrt((weights * (yy - cy) ** 2).sum() / total), 1.0)
    p0 = (amplitude0, cx, cy, w0x, w0y, 0.0, offset0)
    lower = (0.0, x_min, y_min, 0.3, 0.3, -np.pi / 2, -np.inf)
    upper = (np.inf, x_max, y_max, 4.0 * half_size, 4.0 * half_size, np.pi / 2, np.inf)
    popt, _ = curve_fit(gaussian_2d, (xx.ravel(), yy.ravel()), window.ravel(), p0=p0,
                        bounds=(lower, upper), max_nfev=max_nfev)
    A, x0, y0, wx, wy, theta, B = (float(v) for v in popt)
    return dict(x0=x0, y0=y0, wx=wx, wy=wy, theta=theta, amplitude=A, offset=B, peak=A + B,
                window=(x_min, y_min, x_max, y_max))
