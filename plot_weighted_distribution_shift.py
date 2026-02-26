from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


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


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float(np.clip((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n, 0.0, 1.0))


def _quantile_vmax(a: np.ndarray, b: np.ndarray, q: float = 99.5) -> float:
    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if vals.size == 0:
        return 1.0
    vmax = float(np.percentile(vals, q))
    return max(vmax, 1e-9)


def _draw_main_pair(fig, axes_row, left, right, mask, cmap, row_label: str):
    # mask outside support to avoid misleading background colors
    left_m = np.where(mask, left, np.nan)
    right_m = np.where(mask, right, np.nan)
    vmax = _quantile_vmax(left_m, right_m, q=99.5)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f3f3f3")

    im_l = axes_row[0].imshow(left_m.T, origin="lower", cmap=cmap_obj, vmin=0.0, vmax=vmax, interpolation="nearest")
    im_r = axes_row[1].imshow(right_m.T, origin="lower", cmap=cmap_obj, vmin=0.0, vmax=vmax, interpolation="nearest")
    axes_row[0].set_ylabel(row_label, fontsize=11)
    fig.colorbar(im_r, ax=[axes_row[0], axes_row[1]], fraction=0.026, pad=0.02)
    return im_l


def _draw_delta(fig, ax, delta, mask, row_label: str):
    d = np.where(mask, delta, np.nan)
    finite = np.abs(d[np.isfinite(d)])
    lim = float(np.percentile(finite, 99.0)) if finite.size else 1e-6
    lim = max(lim, 1e-9)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f3f3f3")

    im = ax.imshow(d.T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_ylabel(f"{row_label}\nΔ(Weighted-Unweighted)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--base-weight", type=float, default=3.0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--error-quantile", type=float, default=0.55, help="Lower action-error quantile = consistent")
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

    support_mask = (count_u.reshape(args.grid, args.grid) > 0) | (count_w.reshape(args.grid, args.grid) > 0)

    gini_u = _gini(density_u.ravel())
    gini_w = _gini(density_w.ravel())
    gini_delta = gini_u - gini_w  # positive means weighted more uniform

    overall_cons_u = float(cons_u.mean())
    overall_cons_w = float(np.sum(cons_w * w) / np.maximum(np.sum(w), 1e-9))
    cons_delta = overall_cons_w - overall_cons_u

    high_q_u = float(quality_mass_u.sum())
    high_q_w = float(quality_mass_w.sum())
    high_q_delta = high_q_w - high_q_u

    cmap_cons = mcolors.LinearSegmentedColormap.from_list("consistency", ["#ff8c00", "#1ea64b"])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)

    _draw_main_pair(fig, axes[0, :2], density_u, density_w, support_mask, "magma", "Layer 1: Coverage uniformity")
    _draw_delta(fig, axes[0, 2], density_w - density_u, support_mask, "Layer 1")

    _draw_main_pair(fig, axes[1, :2], cons_ratio_u, cons_ratio_w, support_mask, cmap_cons, "Layer 2: Consistent-data ratio")
    _draw_delta(fig, axes[1, 2], cons_ratio_w - cons_ratio_u, support_mask, "Layer 2")

    _draw_main_pair(fig, axes[2, :2], quality_mass_u, quality_mass_w, support_mask, "viridis", "Layer 3: Quality-focused mass")
    _draw_delta(fig, axes[2, 2], quality_mass_w - quality_mass_u, support_mask, "Layer 3")

    axes[0, 0].set_title("Unweighted", fontsize=14, weight="bold")
    axes[0, 1].set_title("Weighted", fontsize=14, weight="bold")
    axes[0, 2].set_title("Difference Map", fontsize=14, weight="bold")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    summary = (
        f"Uniformity (Gini↓): {gini_u:.3f} → {gini_w:.3f} (improve={gini_delta:+.3f})   |   "
        f"Consistency↑: {overall_cons_u:.3f} → {overall_cons_w:.3f} (Δ={cons_delta:+.3f})   |   "
        f"Quality-mass↑: {high_q_u:.4f} → {high_q_w:.4f} (Δ={high_q_delta:+.4f})"
    )
    fig.suptitle(summary, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.984])
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
