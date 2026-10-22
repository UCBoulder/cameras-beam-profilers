"""
Dead pixels and dark spots of a camera: find them in a smoothly lit frame,
repair them for fits, keep them in a JSON file next to the driver.

    found = find_dead_pixels(frames)            # frames: (N, H, W) stack or one frame
    save_dead_pixels(path, found, camera=...)
    found = load_dead_pixels(path)
    fixed = repair(frame, found)                # bad pixels replaced from their surroundings

`found` holds two kinds of bad pixels:

- dead pixels (found["ys"], found["xs"]): single pixels much darker than
  their immediate neighbours; repaired with the mean of the 8 neighbours.
- dark spots (found["spots"], a list of dicts with ys, xs, centre,
  min_fraction): connected patches of 2 or more pixels darker than
  `spot_fraction` (default 70 %) of the smooth local background, such as
  dust or an opaque speck on the sensor window, grown by one pixel to take
  the soft edge; repaired with the mean of a 2-pixel ring around the patch.

Dead-pixel criterion (deliberately strict, so only pixels that are really
dark where the sensor around them is bright and smooth are flagged). With v
the pixel, n_min and n_mean the minimum and mean of its 8 neighbours, and
m_1..m_4 the means of the four opposite neighbour pairs (left/right,
up/down, the two diagonals: on a linear ramp all four equal the value the
pixel should have, so they tolerate gradients and only react to curvature):

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
fails the first two tests.

Dark-spot criterion: the smooth background is a `spot_size` x `spot_size`
median filter of the mean frame (a dark patch covering less than half the
window does not move the median, so spots up to about spot_size / 2 across
vanish from it, while the beam and weak fringes stay); a pixel belongs to a
spot if it is below spot_fraction x background by more than sigma_factor x
the noise there, and the patch has at least 2 pixels. Fringes with more
than 30 % contrast would be flagged too: take the frames with a clean, broad
beam lighting the whole sensor. Pixels on the 1-pixel border are not tested.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

CRITERION = dict(min_drop=0.3, sigma_factor=8.0, smooth_factor=3.0, spot_fraction=0.7, spot_size=25)

_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _neighbours(img: np.ndarray) -> np.ndarray:
    """(8, H-2, W-2) stack of the 8 neighbours of every interior pixel."""
    h, w = img.shape
    return np.stack([img[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] for dy, dx in _OFFSETS])


def _smooth_background(img: np.ndarray, size: int, block: int = 4) -> np.ndarray:
    """
    Smooth level of the lit sensor without the dark spots: median filter of
    `size` x `size` pixels, run on a `block` x `block` block mean of the frame
    for speed (a patch covering less than half the window does not move the
    median). Spots up to about size / 2 across are removed from it.
    """
    h, w = img.shape
    hb, wb = h // block, w // block
    small = img[:hb * block, :wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    n = max(int(round(size / block)) | 1, 3)                     # odd window
    bg_small = ndimage.median_filter(small, size=n, mode="nearest")
    bg = ndimage.zoom(bg_small, block, order=1, mode="nearest")   # back to full resolution
    out = np.empty_like(img)
    out[:bg.shape[0], :bg.shape[1]] = bg[:h, :w]
    out[bg.shape[0]:, :] = out[bg.shape[0] - 1:bg.shape[0], :]    # edge rows / columns lost to the blocks
    out[:, bg.shape[1]:] = out[:, bg.shape[1] - 1:bg.shape[1]]
    return out


def _ring(ys, xs, shape, inner=1, outer=2):
    """Pixels at Chebyshev distance inner+1..outer around the patch (ys, xs), for its repair."""
    h, w = shape
    y0, x0 = max(ys.min() - outer, 0), max(xs.min() - outer, 0)
    y1, x1 = min(ys.max() + outer + 1, h), min(xs.max() + outer + 1, w)
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    mask[ys - y0, xs - x0] = True
    near = ndimage.binary_dilation(mask, iterations=inner)
    far = ndimage.binary_dilation(near, iterations=outer - inner)
    ry, rx = np.nonzero(far & ~near)
    return ry + y0, rx + x0


def find_dead_pixels(frames: np.ndarray, min_drop: float = CRITERION["min_drop"],
                     sigma_factor: float = CRITERION["sigma_factor"],
                     smooth_factor: float = CRITERION["smooth_factor"],
                     spot_fraction: float = CRITERION["spot_fraction"],
                     spot_size: int = CRITERION["spot_size"]) -> dict:
    """
    Dead pixels and dark spots of a (N, H, W) stack of frames, or of one
    (H, W) frame (see the module docstring); coordinates are array indices.
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
    h, w = img.shape
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

    # dark spots: patches below spot_fraction of the local background, at least 2 pixels
    background = _smooth_background(img, spot_size)
    bg_sigma = np.sqrt(sigma ** 2 + shot * np.maximum(background, 0) / n_frames)
    low = np.zeros((h, w), bool)
    low[1:-1, 1:-1] = ((img < spot_fraction * background) & (background - img > sigma_factor * bg_sigma))[1:-1, 1:-1]
    labels, n = ndimage.label(low, structure=np.ones((3, 3)))
    spots = []
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        patch = labels[box] == k
        if patch.sum() < 2:
            continue
        # work on the bounding box grown by 1 px: take the soft edge, stay below the background
        y0, x0 = max(box[0].start - 1, 0), max(box[1].start - 1, 0)
        y1, x1 = min(box[0].stop + 1, h), min(box[1].stop + 1, w)
        local = np.zeros((y1 - y0, x1 - x0), bool)
        local[box[0].start - y0:box[0].stop - y0, box[1].start - x0:box[1].stop - x0] = patch
        grown = ndimage.binary_dilation(local) & (img[y0:y1, x0:x1] < background[y0:y1, x0:x1])
        py, px = np.nonzero(grown)
        py, px = py + y0, px + x0
        keep = (py > 0) & (py < h - 1) & (px > 0) & (px < w - 1)
        py, px = py[keep], px[keep]
        core = img[y0:y1, x0:x1][local] / background[y0:y1, x0:x1][local]
        spots.append(dict(ys=py, xs=px, centre=(float(py.mean()), float(px.mean())),
                          n_pixels=int(len(py)), min_fraction=float(core.min())))
    spots.sort(key=lambda s: -s["n_pixels"])

    found = dict(ys=ys, xs=xs, values=img[ys, xs], surroundings=n_mean[ys - 1, xs - 1], spots=spots, shape=(h, w),
                 noise_sigma=sigma, shot_noise_per_count=shot,
                 criterion=dict(min_drop=min_drop, sigma_factor=sigma_factor, smooth_factor=smooth_factor,
                                spot_fraction=spot_fraction, spot_size=spot_size, n_frames=n_frames))
    return _with_rings(found)


def all_bad_pixels(found: dict) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of every dead pixel and dark-spot pixel in `found`."""
    ys = [np.asarray(found["ys"], int)] + [np.asarray(s["ys"], int) for s in found.get("spots", [])]
    xs = [np.asarray(found["xs"], int)] + [np.asarray(s["xs"], int) for s in found.get("spots", [])]
    return np.concatenate(ys), np.concatenate(xs)


def repair(frame: np.ndarray, found: dict, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Float copy of `frame` with the dead pixels replaced by the mean of their 8
    neighbours and each dark spot by the mean of a ring around it. `offset`
    = (y, x) of the frame's first pixel on the sensor (a readout region).
    """
    img = np.asarray(frame, dtype=float).copy()
    h, w = img.shape
    oy, ox = offset
    ys, xs = np.asarray(found["ys"], int) - oy, np.asarray(found["xs"], int) - ox
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    ys, xs = ys[inside], xs[inside]
    if len(ys):
        total, count = np.zeros(len(ys)), np.zeros(len(ys))
        for dy, dx in _OFFSETS:
            y, x = ys + dy, xs + dx
            ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            total[ok] += img[y[ok], x[ok]]
            count[ok] += 1
        img[ys, xs] = total / np.maximum(count, 1)
    for spot in found.get("spots", []):
        sy, sx = np.asarray(spot["ys"], int) - oy, np.asarray(spot["xs"], int) - ox
        ry, rx = np.asarray(spot["ring_ys"], int) - oy, np.asarray(spot["ring_xs"], int) - ox
        ok_s = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        ok_r = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
        if ok_s.any() and ok_r.any():
            img[sy[ok_s], sx[ok_s]] = img[ry[ok_r], rx[ok_r]].mean()
    return img


def _with_rings(found: dict) -> dict:
    """Add the repair ring (ring_ys, ring_xs) to every spot that lacks it."""
    shape = tuple(found["shape"])
    for spot in found.get("spots", []):
        if "ring_ys" not in spot:
            spot["ring_ys"], spot["ring_xs"] = _ring(np.asarray(spot["ys"], int), np.asarray(spot["xs"], int), shape)
    return found


def save_dead_pixels(path: str | Path, found: dict, **info) -> Path:
    """Write the result of find_dead_pixels() (plus any `info`, e.g. camera, serial, exposure) as JSON."""
    path = Path(path)
    data = dict(info)
    data.update(
        found=datetime.now().isoformat(timespec="seconds"),
        shape=[int(n) for n in found["shape"]],
        criterion=found["criterion"],
        noise_sigma=round(found["noise_sigma"], 3),
        shot_noise_per_count=round(found["shot_noise_per_count"], 4),
        n_dead=int(len(found["ys"])),
        n_spots=len(found["spots"]),
        pixel_format="[y, x, value, mean of the 8 neighbours] in sensor pixels, (0, 0) = top-left",
        pixels=[[int(y), int(x), round(float(v), 2), round(float(s), 2)]
                for y, x, v, s in zip(found["ys"], found["xs"], found["values"], found["surroundings"])],
        spot_format="centre [y, x], n_pixels, min_fraction = darkest pixel / local background, pixels [[y, x], ...]",
        spots=[dict(centre=[round(c, 1) for c in s["centre"]], n_pixels=s["n_pixels"],
                    min_fraction=round(s["min_fraction"], 3),
                    pixels=[[int(y), int(x)] for y, x in zip(s["ys"], s["xs"])]) for s in found["spots"]],
    )
    path.write_text(json.dumps(data, indent=1))
    return path


def load_dead_pixels(path: str | Path) -> dict:
    """The `found` dict (ys, xs, spots with repair rings, info) from a file written by save_dead_pixels()."""
    data = json.loads(Path(path).read_text())
    pixels = np.array(data.get("pixels", []), dtype=float).reshape(-1, 4)
    spots = []
    for s in data.get("spots", []):
        p = np.array(s["pixels"], dtype=int).reshape(-1, 2)
        spots.append(dict(ys=p[:, 0], xs=p[:, 1], centre=tuple(s["centre"]), n_pixels=s["n_pixels"],
                          min_fraction=s["min_fraction"]))
    found = dict(ys=pixels[:, 0].astype(int), xs=pixels[:, 1].astype(int), spots=spots,
                 shape=tuple(data.get("shape", (0, 0))),
                 info={k: v for k, v in data.items() if k not in ("pixels", "spots")})
    return _with_rings(found) if found["shape"] != (0, 0) else found
