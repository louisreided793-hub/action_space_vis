from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


GRAY = (0.75, 0.75, 0.75, 1.0)


def _load_precomputed(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    z = data["z2d"].astype(np.float32)
    actions = data["action_latent"].astype(np.float32)
    demo_idx = data["demo_idx"].astype(np.int64)
    return z, actions, demo_idx


def _load_weights(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bin_points(z: np.ndarray, grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad = 1e-6
    x_min, x_max = z[:, 0].min() - pad, z[:, 0].max() + pad
    y_min, y_max = z[:, 1].min() - pad, z[:, 1].max() + pad
    x_edges = np.linspace(x_min, x_max, grid + 1)
    y_edges = np.linspace(y_min, y_max, grid + 1)

    ix = np.searchsorted(x_edges, z[:, 0], side="right") - 1
    iy = np.searchsorted(y_edges, z[:, 1], side="right") - 1
    valid = (ix >= 0) & (ix < grid) & (iy >= 0) & (iy < grid)
    valid_idx = np.nonzero(valid)[0]
    return ix[valid], iy[valid], valid_idx


def _mean_action_per_cell(flat: np.ndarray, actions: np.ndarray, weights: np.ndarray, num_cells: int) -> np.ndarray:
    sum_w = np.bincount(flat, weights=weights, minlength=num_cells)
    sum_w = np.maximum(sum_w, 1e-9)
    mean = np.zeros((num_cells, actions.shape[1]), dtype=np.float64)
    for d in range(actions.shape[1]):
        sums = np.bincount(flat, weights=weights * actions[:, d], minlength=num_cells)
        mean[:, d] = sums / sum_w
    return mean.astype(np.float32)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            sx0 = max(0, -dx)
            sx1 = min(mask.shape[0], mask.shape[0] - dx)
            sy0 = max(0, -dy)
            sy1 = min(mask.shape[1], mask.shape[1] - dy)

            tx0 = max(0, dx)
            tx1 = min(mask.shape[0], mask.shape[0] + dx)
            ty0 = max(0, dy)
            ty1 = min(mask.shape[1], mask.shape[1] + dy)

            out[tx0:tx1, ty0:ty1] |= mask[sx0:sx1, sy0:sy1]
    return out


def _norm01(x: np.ndarray) -> np.ndarray:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x)
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _quality_map(cons_ratio: np.ndarray, weight_mass: np.ndarray) -> np.ndarray:
    c = _norm01(cons_ratio)
    w = _norm01(weight_mass)
    # high consistency + high weight => green; otherwise orange
    return np.clip(c * w, 0.0, 1.0)


def _render_rgba(value01: np.ndarray, visible_mask: np.ndarray, cmap: mcolors.Colormap) -> np.ndarray:
    rgba = cmap(value01)
    rgba = np.asarray(rgba, dtype=np.float32)
    rgba[~visible_mask] = np.array(GRAY, dtype=np.float32)
    return rgba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--base-weight", type=float, default=3.0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--error-quantile", type=float, default=0.55, help="Lower action-error quantile = consistent")
    ap.add_argument("--mask-dilate", type=int, default=2, help="Action-space mask dilation radius in cells")
    args = ap.parse_args()

    out_fig = Path(args.out_fig).expanduser().resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    weight_map = _load_weights(args.weights)
    z, actions, demo_idx = _load_precomputed(args.precomputed)

    max_demo = int(demo_idx.max()) if demo_idx.size else -1
    demo_weights = np.empty((max_demo + 1,), dtype=np.float32) if max_demo >= 0 else np.zeros((0,), dtype=np.float32)
    for idx in range(max_demo + 1):
        key = f"demo_{idx}"
        if key not in weight_map:
            raise KeyError(f"Missing weight for {key}")
        demo_weights[idx] = float(weight_map[key])

    raw_weights = demo_weights[demo_idx] if demo_idx.size else np.zeros((0,), dtype=np.float32)
    scaled_weights = args.base_weight + (raw_weights - args.base_weight) * args.weight_scale

    ix, iy, valid_idx = _bin_points(z, args.grid)
    actions = actions[valid_idx]
    w = scaled_weights[valid_idx]

    flat = ix + iy * args.grid
    num_cells = args.grid * args.grid

    count_u = np.bincount(flat, minlength=num_cells).astype(np.float64)
    count_w = np.bincount(flat, weights=w, minlength=num_cells).astype(np.float64)

    weight_mass_u = (count_u / np.maximum(count_u.sum(), 1.0)).reshape(args.grid, args.grid)
    weight_mass_w = (count_w / np.maximum(count_w.sum(), 1.0)).reshape(args.grid, args.grid)

    mean_u = _mean_action_per_cell(flat, actions, np.ones_like(w), num_cells)
    mean_w = _mean_action_per_cell(flat, actions, w, num_cells)

    err_u = np.sum((actions - mean_u[flat]) ** 2, axis=1)
    err_w = np.sum((actions - mean_w[flat]) ** 2, axis=1)
    ref = np.concatenate([err_u, err_w])
    threshold = float(np.quantile(ref, np.clip(args.error_quantile, 0.01, 0.99)))

    cons_u = (err_u <= threshold).astype(np.float32)
    cons_w = (err_w <= threshold).astype(np.float32)

    cons_num_u = np.bincount(flat, weights=cons_u, minlength=num_cells)
    cons_den_u = np.maximum(np.bincount(flat, minlength=num_cells), 1.0)
    cons_ratio_u = (cons_num_u / cons_den_u).reshape(args.grid, args.grid)

    cons_num_w = np.bincount(flat, weights=cons_w * w, minlength=num_cells)
    cons_den_w = np.maximum(np.bincount(flat, weights=w, minlength=num_cells), 1e-9)
    cons_ratio_w = (cons_num_w / cons_den_w).reshape(args.grid, args.grid)

    support = ((count_u + count_w) > 0).reshape(args.grid, args.grid)
    visible_mask = _dilate_mask(support, args.mask_dilate)

    quality_u = _quality_map(cons_ratio_u, weight_mass_u)
    quality_w = _quality_map(cons_ratio_w, weight_mass_w)

    # Orange -> Green
    cmap = mcolors.LinearSegmentedColormap.from_list("quality", ["#ff8c00", "#1ea64b"])
    rgba_u = _render_rgba(quality_u, visible_mask, cmap)
    rgba_w = _render_rgba(quality_w, visible_mask, cmap)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), sharex=True, sharey=True)
    axes[0].imshow(rgba_u.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    axes[1].imshow(rgba_w.transpose(1, 0, 2), origin="lower", interpolation="nearest")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(out_fig, dpi=800, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
