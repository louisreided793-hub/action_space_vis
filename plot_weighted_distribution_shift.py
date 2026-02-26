from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


ORANGE = "#ff8c00"
GREEN = "#1ea64b"


def _load_precomputed(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    z = data["z2d"].astype(np.float32)
    actions = data["action_latent"].astype(np.float32)
    demo_idx = data["demo_idx"].astype(np.int64)
    return z, actions, demo_idx


def _load_weights(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bin_points(z: np.ndarray, grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pad = 1e-6
    x_min, x_max = z[:, 0].min() - pad, z[:, 0].max() + pad
    y_min, y_max = z[:, 1].min() - pad, z[:, 1].max() + pad

    x_edges = np.linspace(x_min, x_max, grid + 1)
    y_edges = np.linspace(y_min, y_max, grid + 1)

    ix = np.searchsorted(x_edges, z[:, 0], side="right") - 1
    iy = np.searchsorted(y_edges, z[:, 1], side="right") - 1

    valid = (ix >= 0) & (ix < grid) & (iy >= 0) & (iy < grid)
    valid_idx = np.nonzero(valid)[0]
    return ix[valid], iy[valid], valid_idx, x_edges, y_edges, valid


def _mean_action_per_cell(flat: np.ndarray, actions: np.ndarray, weights: np.ndarray, num_cells: int) -> np.ndarray:
    sum_w = np.bincount(flat, weights=weights, minlength=num_cells)
    sum_w = np.maximum(sum_w, 1e-9)
    mean = np.zeros((num_cells, actions.shape[1]), dtype=np.float64)
    for d in range(actions.shape[1]):
        sums = np.bincount(flat, weights=weights * actions[:, d], minlength=num_cells)
        mean[:, d] = sums / sum_w
    return mean.astype(np.float32)


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(np.clip(g, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--base-weight", type=float, default=3.0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--error-quantile", type=float, default=0.65, help="Lower action-error quantile = consistent")
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

    ix, iy, valid_idx, _, _, _ = _bin_points(z, args.grid)
    z = z[valid_idx]
    actions = actions[valid_idx]
    w = scaled_weights[valid_idx]

    flat = ix + iy * args.grid
    num_cells = args.grid * args.grid

    count_u = np.bincount(flat, minlength=num_cells).astype(np.float64)
    count_w = np.bincount(flat, weights=w, minlength=num_cells).astype(np.float64)

    density_u = (count_u / np.maximum(count_u.sum(), 1.0)).reshape(args.grid, args.grid)
    density_w = (count_w / np.maximum(count_w.sum(), 1.0)).reshape(args.grid, args.grid)

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

    quality_mass_u = density_u * cons_ratio_u
    quality_mass_w = density_w * cons_ratio_w

    gini_u = _gini(density_u.ravel())
    gini_w = _gini(density_w.ravel())

    overall_cons_u = float(cons_u.mean())
    overall_cons_w = float(np.sum(cons_w * w) / np.maximum(np.sum(w), 1e-9))

    cmap_density = "magma"
    cmap_cons = mcolors.LinearSegmentedColormap.from_list("consistency", [ORANGE, GREEN])
    cmap_quality = "viridis"

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
    grids = [
        (density_u, density_w, cmap_density, "Layer 1: Coverage uniformity (cell density)"),
        (cons_ratio_u, cons_ratio_w, cmap_cons, "Layer 2: Consistent-data ratio (orange→green)"),
        (quality_mass_u, quality_mass_w, cmap_quality, "Layer 3: Weight focus on high-quality regions"),
    ]

    col_titles = [
        f"Unweighted\nGini={gini_u:.3f}",
        f"Weighted\nGini={gini_w:.3f}",
    ]

    for r, (left, right, cmap, row_title) in enumerate(grids):
        shared_vmax = float(max(np.max(left), np.max(right), 1e-9))
        im_l = axes[r, 0].imshow(left.T, origin="lower", cmap=cmap, vmin=0.0, vmax=shared_vmax, interpolation="nearest")
        im_r = axes[r, 1].imshow(right.T, origin="lower", cmap=cmap, vmin=0.0, vmax=shared_vmax, interpolation="nearest")
        axes[r, 0].set_ylabel(row_title, fontsize=11)
        fig.colorbar(im_r, ax=axes[r, :], fraction=0.025, pad=0.02)

    for c in range(2):
        axes[0, c].set_title(col_titles[c], fontsize=14, weight="bold")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    summary = (
        f"Consistent ratio: {overall_cons_u:.3f} → {overall_cons_w:.3f}   |   "
        f"Coverage Gini: {gini_u:.3f} → {gini_w:.3f} (lower is more uniform)"
    )
    fig.suptitle(summary, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
