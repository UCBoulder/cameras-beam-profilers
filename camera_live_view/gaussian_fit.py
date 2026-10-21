"""
2D Gaussian fit of the brightest spot in a frame, for the live viewer.

    fit = fit_gaussian_2d(frame, half_size=30)
    fit["x0"], fit["y0"]      centre in array coordinates (pixel i centred at i)
    fit["wx"], fit["wy"]      1/e² radii (px) along the ellipse axes, fit["theta"] their rotation

Cheap enough for ten fits per second whatever the window size:

- the spot is located on a 4 × 4 block mean of the frame (a hot pixel cannot
  win), then the window (2 half_size + 1)² around the brightest pixel is cut;
- the window is binned b × b so that at most `max_pixels` values enter the
  least squares (b = 1 up to ±30 px, b = 16 for the whole 1280 × 1024 sensor);
  the coordinates stay in sensor pixels, so nothing is rescaled afterwards;
- start values: centre from the centroid above half maximum, width from the
  area above half maximum, bounded least squares with Jacobian scaling;
- a spot narrower than 4 b pixels (a small spot in a big window) is refitted
  in a window of ±4 w around the first result, with finer binning, until it
  is resolved: a wide window never costs resolution on a small spot.
"""

import numpy as np
from scipy.optimize import curve_fit

MAX_PIXELS = 4000          # values entering one least squares (~10 ms)


def gaussian_2d(coords, A, x0, y0, wx, wy, theta, B):
    """Rotated 2D Gaussian with 1/e² radii wx, wy (beam-optics convention) on a constant B."""
    x, y = coords
    dx, dy = x - x0, y - y0
    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)
    return A * np.exp(-2.0 * (xp ** 2 / wx ** 2 + yp ** 2 / wy ** 2)) + B


def find_spot(img: np.ndarray, block: int = 4) -> tuple[int, int]:
    """(iy, ix) of the brightest pixel inside the brightest `block` × `block` mean."""
    h, w = img.shape
    hb, wb = h // block, w // block
    coarse = img[:hb * block, :wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    by, bx = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    y0, x0 = max((by - 1) * block, 0), max((bx - 1) * block, 0)
    patch = img[y0:(by + 2) * block, x0:(bx + 2) * block]
    py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return y0 + py, x0 + px


def fit_window(img: np.ndarray, ix: int, iy: int, half_size: int, max_pixels: int = MAX_PIXELS,
               max_nfev: int = 100) -> dict:
    """Fit gaussian_2d() to the (binned) window of ±half_size around pixel (ix, iy)."""
    h_img, w_img = img.shape
    x_min, x_max = max(ix - half_size, 0), min(ix + half_size + 1, w_img)
    y_min, y_max = max(iy - half_size, 0), min(iy + half_size + 1, h_img)
    window = img[y_min:y_max, x_min:x_max]

    b = max(int(np.ceil(np.sqrt(window.size / max_pixels))), 1)
    if b > 1:                                      # block mean; coordinates = centres of the blocks
        ny, nx = window.shape[0] // b, window.shape[1] // b
        window = window[:ny * b, :nx * b].reshape(ny, b, nx, b).mean(axis=(1, 3))
        yy, xx = np.meshgrid(y_min + b * np.arange(ny) + (b - 1) / 2,
                             x_min + b * np.arange(nx) + (b - 1) / 2, indexing="ij")
    else:
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(float)

    # start values: offset from the low percentile; centre and width from the pixels above half maximum
    # (a Gaussian covers 1.089 wx wy pixels above half maximum)
    offset0 = float(np.percentile(window, 10))
    amplitude0 = float(window.max() - offset0)
    above = window > offset0 + 0.5 * amplitude0
    n_above = int(above.sum())
    if amplitude0 <= 0 or n_above == 0:
        raise RuntimeError("no spot in the frame")
    cx, cy = float(xx[above].mean()), float(yy[above].mean())
    w0 = max(b * np.sqrt(n_above / 1.089), 1.0)
    if b > 1 and n_above < 3:
        # unresolved at this binning: return the estimate, the caller refines around it
        return dict(x0=cx, y0=cy, wx=w0, wy=w0, theta=0.0, amplitude=amplitude0, offset=offset0,
                    peak=amplitude0 + offset0, window=(x_min, y_min, x_max, y_max), binning=b)
    p0 = (amplitude0, cx, cy, w0, w0, 0.0, offset0)
    lower = (0.0, x_min, y_min, 0.3 * b, 0.3 * b, -np.pi / 2, -np.inf)
    upper = (np.inf, x_max, y_max, 4.0 * half_size, 4.0 * half_size, np.pi / 2, np.inf)
    popt, _ = curve_fit(gaussian_2d, (xx.ravel(), yy.ravel()), window.ravel(), p0=p0,
                        bounds=(lower, upper), max_nfev=max_nfev, x_scale="jac")
    A, x0, y0, wx, wy, theta, B = (float(v) for v in popt)
    return dict(x0=x0, y0=y0, wx=wx, wy=wy, theta=theta, amplitude=A, offset=B, peak=A + B,
                window=(int(x_min), int(y_min), int(x_max), int(y_max)), binning=b)


def fit_gaussian_2d(frame: np.ndarray, half_size: int = 30, max_pixels: int = MAX_PIXELS) -> dict:
    """
    Fit gaussian_2d() to the spot around the brightest pixel of `frame`.

    Returns a dict with x0, y0, wx, wy, theta, amplitude, offset, peak (A + B),
    window (x_min, y_min, x_max, y_max) and binning (b, 1 = unbinned); raises
    RuntimeError if the fit does not converge.
    """
    img = np.asarray(frame, dtype=float)
    iy, ix = find_spot(img)
    half = half_size
    for _ in range(4):
        fit = fit_window(img, ix, iy, half, max_pixels)
        w_max = max(fit["wx"], fit["wy"])
        if fit["binning"] == 1 or w_max >= 4 * fit["binning"]:
            break
        # small spot in a big window: refit around it with a window matched to its size
        half = max(int(np.ceil(4 * w_max)), 5)
        ix, iy = int(round(fit["x0"])), int(round(fit["y0"]))
    return fit
