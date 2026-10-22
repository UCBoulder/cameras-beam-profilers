"""
Dead pixels of a camera: find them in a smoothly lit frame, repair them for
fits, keep them in a JSON file next to the driver.

    found = find_dead_pixels(frames)            # frames: (N, H, W) stack or one frame; dict with ys, xs, noise, criterion
    save_dead_pixels(path, found, camera=...)
    ys, xs, info = load_dead_pixels(path)
    fixed = repair(frame, ys, xs)               # dead pixels replaced by the mean of their 8 neighbours

Criterion (deliberately strict, so only pixels that are really dark where
the sensor around them is bright and smooth are flagged). With v the pixel,
n_min and n_mean the minimum and mean of its 8 neighbours, and m_1..m_4 the
means of the four opposite neighbour pairs (left/right, up/down, the two
diagonals: on a linear ramp all four equal the value the pixel should have,
so they tolerate gradients and only react to curvature):

    n_min - v > sigma_factor * sigma_pix        the pixel is the darkest of its 3 x 3 by far more than the noise (8 sigma)
    drop > min_drop * n_mean                    with drop = min(m_k) - v: much darker than any linear prediction (30 %)
    drop > smooth_factor * (max(m_k) - min(m_k))   and the surroundings are smooth compared to the drop (3 x)

sigma_pix = sqrt(sigma^2 + g * n_mean / N) is the noise expected at that
pixel of the N-frame mean: the robust spatial noise of the mean frame
(sigma = 1.4826 x median |v - n_mean|, which includes quantisation and fixed
pattern) plus shot noise at the local level. The shot-noise constant g
(counts of variance per count of signal) is measured from the stack as the
median of the frame-to-frame variance over the mean, so it is right for any
gain setting; from a single frame it is taken as 1, which is pessimistic. An
edge fails the smoothness test (the pair across it differs from the pair
along it by half the step); a fringe of a few pixels period passes it, so a
dead pixel on a fringed beam is still found; a dark pixel in a dark region
fails the first two tests. Take the frame with a broad beam lighting the
whole sensor, averaged over a few frames; pixels on the 1-pixel border are
not tested.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

CRITERION = dict(min_drop=0.3, sigma_factor=8.0, smooth_factor=3.0)

_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _neighbours(img: np.ndarray) -> np.ndarray:
    """(8, H-2, W-2) stack of the 8 neighbours of every interior pixel."""
    h, w = img.shape
    return np.stack([img[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] for dy, dx in _OFFSETS])


def find_dead_pixels(frames: np.ndarray, min_drop: float = CRITERION["min_drop"],
                     sigma_factor: float = CRITERION["sigma_factor"],
                     smooth_factor: float = CRITERION["smooth_factor"]) -> dict:
    """
    Dead pixels of a (N, H, W) stack of frames, or of one (H, W) frame (see the
    module docstring); coordinates are array indices.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 3:
        n_frames = frames.shape[0]
        img = frames.mean(axis=0)
        var = frames.var(axis=0, ddof=1) if n_frames > 1 else np.zeros_like(img)
        lit = img > np.percentile(img, 50)
        shot = float(np.median(var[lit] / np.maximum(img[lit], 1e-9))) if lit.any() and n_frames > 1 else 1.0
    else:
        n_frames, img, shot = 1, frames, 1.0
    v = img[1:-1, 1:-1]
    nb = _neighbours(img)
    n_min, n_mean = nb.min(axis=0), nb.mean(axis=0)
    sigma = 1.4826 * float(np.median(np.abs(v - n_mean)))
    pairs = np.stack([(nb[k] + nb[7 - k]) / 2 for k in range(4)])   # _OFFSETS[k] and _OFFSETS[7 - k] are opposite
    drop = pairs.min(axis=0) - v
    curvature = pairs.max(axis=0) - pairs.min(axis=0)
    sigma_pix = np.sqrt(sigma ** 2 + shot * np.maximum(n_mean, 0) / n_frames)
    dead = (n_min - v > sigma_factor * sigma_pix) & (drop > min_drop * n_mean) & (drop > smooth_factor * curvature)
    ys, xs = np.nonzero(dead)
    ys, xs = ys + 1, xs + 1
    return dict(ys=ys, xs=xs, values=img[ys, xs], surroundings=n_mean[ys - 1, xs - 1], noise_sigma=sigma,
                shot_noise_per_count=shot,
                criterion=dict(min_drop=min_drop, sigma_factor=sigma_factor, smooth_factor=smooth_factor,
                               n_frames=n_frames))


def repair(frame: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Float copy of `frame` with the pixels (ys, xs) replaced by the mean of their 8 neighbours."""
    img = np.asarray(frame, dtype=float).copy()
    if len(ys) == 0:
        return img
    h, w = img.shape
    ys, xs = np.asarray(ys), np.asarray(xs)
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    ys, xs = ys[inside], xs[inside]
    total, count = np.zeros(len(ys)), np.zeros(len(ys))
    for dy, dx in _OFFSETS:
        y, x = ys + dy, xs + dx
        ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        total[ok] += img[y[ok], x[ok]]
        count[ok] += 1
    img[ys, xs] = total / np.maximum(count, 1)
    return img


def save_dead_pixels(path: str | Path, found: dict, **info) -> Path:
    """Write the result of find_dead_pixels() (plus any `info`, e.g. camera, serial, exposure) as JSON."""
    path = Path(path)
    data = dict(info)
    data.update(
        found=datetime.now().isoformat(timespec="seconds"),
        criterion=found["criterion"],
        noise_sigma=round(found["noise_sigma"], 3),
        shot_noise_per_count=round(found["shot_noise_per_count"], 4),
        n_dead=int(len(found["ys"])),
        pixel_format="[y, x, value, mean of the 8 neighbours] in sensor pixels, (0, 0) = top-left",
        pixels=[[int(y), int(x), round(float(v), 2), round(float(s), 2)]
                for y, x, v, s in zip(found["ys"], found["xs"], found["values"], found["surroundings"])],
    )
    path.write_text(json.dumps(data, indent=1))
    return path


def load_dead_pixels(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(ys, xs, info) from a file written by save_dead_pixels()."""
    data = json.loads(Path(path).read_text())
    pixels = np.array(data.get("pixels", []), dtype=float).reshape(-1, 4)
    info = {k: v for k, v in data.items() if k != "pixels"}
    return pixels[:, 0].astype(int), pixels[:, 1].astype(int), info
