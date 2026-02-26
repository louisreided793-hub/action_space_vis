"""
plot_weighted_distribution_shift.py

Visualize how per-demo weights reshape state coverage and action consistency from precomputed data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def _bin_points(z: np.ndarray, grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pad = 1e-6
    x_min, x_max = z[:, 0].min(), z[:, 0].max()
    y_min, y_max = z[:, 1].min(), z[:, 1].max()
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    x_edges = np.linspace(x_min, x_max, grid + 1)
    y_edges = np.linspace(y_min, y_max, grid + 1)

    ix = np.searchsorted(x_edges, z[:, 0], side="right") - 1
    iy = np.searchsorted(y_edges, z[:, 1], side="right") - 1

    valid = (ix >= 0) & (ix < grid) & (iy >= 0) & (iy < grid)
    valid_idx = np.nonzero(valid)[0]
    ix = ix[valid]
    iy = iy[valid]
    return ix, iy, x_edges, y_edges, valid_idx


def _cell_stats(flat: np.ndarray, actions: np.ndarray, weights: np.ndarray, num_cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_w = np.bincount(flat, weights=weights, minlength=num_cells)
    sum_a = np.zeros((num_cells, actions.shape[1]), dtype=np.float64)
    for d in range(actions.shape[1]):
        np.add.at(sum_a[:, d], flat, weights * actions[:, d])
    sum_a2 = np.bincount(flat, weights=weights * np.sum(actions ** 2, axis=1), minlength=num_cells)
    return sum_w, sum_a, sum_a2


def _size_from_weight(values: np.ndarray, base_weight: float, base_area: float) -> np.ndarray:
    """Map weights to marker areas using linear area scaling (points^2)."""
    eps = 1e-9
    scale = np.maximum(values, 0.0) / max(base_weight, eps)
    return (base_area * scale).astype(np.float32)


def _normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.full_like(values, 0.5, dtype=np.float32)
    scaled = (values - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _percentile_rank(values: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    if ref_sorted.size <= 1:
        return np.full_like(values, 0.5, dtype=np.float32)
    idx = np.searchsorted(ref_sorted, values, side="left")
    return np.clip(idx / (ref_sorted.size - 1), 0.0, 1.0).astype(np.float32)


def _scale_to_range(values: np.ndarray, vmin: float, vmax: float, out_min: float, out_max: float) -> np.ndarray:
    scaled = _normalize(values, vmin, vmax)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32)


def _apply_saturation(rgb: np.ndarray, sat_scale: np.ndarray, value_scale: float) -> np.ndarray:
    hsv = mcolors.rgb_to_hsv(rgb)
    # Use sat_scale directly for saturation; set value from value_scale.
    hsv[..., 1] = np.clip(sat_scale, 0.0, 1.0)
    hsv[..., 2] = np.clip(value_scale, 0.0, 1.0)
    return mcolors.hsv_to_rgb(hsv).astype(np.float32)


DEFAULT_COLORS = ["#ff7a00", "#ffd500", "#00c853"]
BLEND_ALPHA = 1.0


def _scatter_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
    alpha_main: float,
) -> None:
    use_cmap = not (isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] in (3, 4))
    scatter_kwargs = dict(
        c=c,
        linewidths=0,
        edgecolors="none",
        antialiased=True,
        rasterized=True,
    )
    if use_cmap:
        scatter_kwargs.update(cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha_main,
        zorder=2,
        **scatter_kwargs,
    )


def _load_precomputed(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    z = data["z2d"].astype(np.float32)
    actions = data["action_latent"].astype(np.float32)
    demo_idx = data["demo_idx"].astype(np.int64)
    return z, actions, demo_idx


def _load_weights(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)


def _parse_colors(value: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Need at least 2 colors for gradient.")
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precomputed", required=True, help="Path to precomputed z2d/action_latent/demo_idx npz.")
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--grid", type=int, default=80)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--base-weight", type=float, default=3.0)
    ap.add_argument("--base-dot-size", type=float, default=64.0, help="Marker area (points^2) at base weight.")
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--clip", action="store_true", help="Clip SD saturation range to 0.5-99.5th percentile.")
    ap.add_argument(
        "--ac-colors",
        default=",".join(DEFAULT_COLORS),
        help="Comma-separated list of hex colors for AC gradient.",
    )
    ap.add_argument(
        "--ac-mode",
        choices=["linear", "tanh", "percentile", "percentile_linear", "percentile_tanh", "minmax"],
        default="linear",
        help="AC normalization mode.",
    )
    ap.add_argument("--value-scale", type=float, default=1.0, help="HSV value multiplier for AC colors.")
    ap.add_argument("--ac-scale", type=float, default=1.0, help="Tanh scale for AC; >1 flattens extremes.")
    ap.add_argument("--no-ac", action="store_true", help="Disable AC overlay; use constant hue with SD saturation.")
    args = ap.parse_args()

    blend_alpha = BLEND_ALPHA
    ac_colors = _parse_colors(args.ac_colors)
    ac_cmap = mcolors.LinearSegmentedColormap.from_list("ac_gradient", list(ac_colors))

    out_fig = Path(args.out_fig).expanduser().resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    weight_map = _load_weights(args.weights)
    z, actions, demo_idx = _load_precomputed(args.precomputed)

    if demo_idx.size:
        max_demo = int(demo_idx.max())
        demo_weights = np.empty((max_demo + 1,), dtype=np.float32)
        for idx in range(max_demo + 1):
            w = weight_map.get(f"demo_{idx}")
            if w is None:
                raise KeyError(f"Missing weight for demo_{idx}")
            demo_weights[idx] = float(w)
        weights = demo_weights[demo_idx]
    else:
        weights = np.zeros((0,), dtype=np.float32)

    weights = args.base_weight + (weights - args.base_weight) * args.weight_scale

    ix, iy, x_edges, y_edges, valid_idx = _bin_points(z, args.grid)
    flat = ix + iy * args.grid
    num_cells = args.grid * args.grid

    counts = np.bincount(flat, minlength=num_cells).astype(np.float64)
    wcounts = np.bincount(flat, weights=weights, minlength=num_cells).astype(np.float64)
    counts_grid = counts.reshape(args.grid, args.grid)
    wcounts_grid = wcounts.reshape(args.grid, args.grid)

    total_count = max(counts.sum(), 1.0)
    total_wcount = max(wcounts.sum(), 1.0)
    density_grid = counts_grid / total_count
    wdensity_grid = wcounts_grid / total_wcount

    z_valid = z[valid_idx]
    weights_valid = weights[valid_idx]

    # SD sizes: use raw demo weight by area
    base_area = float(args.base_dot_size)
    size_unw = np.full_like(weights_valid, base_area, dtype=np.float32)
    size_w = _size_from_weight(weights_valid, args.base_weight, base_area)

    density_per_point = density_grid[ix, iy]
    wdensity_per_point = wdensity_grid[ix, iy]
    sd_vals = np.concatenate([density_grid[np.isfinite(density_grid)], wdensity_grid[np.isfinite(wdensity_grid)]])
    if sd_vals.size:
        if args.clip:
            sd_vmin, sd_vmax = np.percentile(sd_vals, [0.5, 99.5])
        else:
            sd_vmin, sd_vmax = np.min(sd_vals), np.max(sd_vals)
    else:
        sd_vmin, sd_vmax = 0.0, 1.0

    sat_min = 0.3
    sat_max = 1.0
    sat_unw = _scale_to_range(density_per_point, sd_vmin, sd_vmax, sat_min, sat_max)
    sat_w = _scale_to_range(wdensity_per_point, sd_vmin, sd_vmax, sat_min, sat_max)

    # Action consistency shift
    sum_w_u, sum_a_u, sum_a2_u = _cell_stats(flat, actions, np.ones_like(weights), num_cells)
    sum_w_w, sum_a_w, sum_a2_w = _cell_stats(flat, actions, weights, num_cells)

    def _consistency(sum_w, sum_a, sum_a2):
        sum_w_safe = np.maximum(sum_w, 1e-9)
        mean_a = sum_a / sum_w_safe[:, None]
        trace = sum_a2 / sum_w_safe - np.sum(mean_a ** 2, axis=1)
        trace = np.maximum(trace, 1e-12)
        return 1.0 / trace, mean_a

    cons_u, mean_a_u = _consistency(sum_w_u, sum_a_u, sum_a2_u)
    cons_w, mean_a_w = _consistency(sum_w_w, sum_a_w, sum_a2_w)

    cons_u_grid = cons_u.reshape(args.grid, args.grid)
    cons_w_grid = cons_w.reshape(args.grid, args.grid)

    if args.no_ac:
        base_rgb = np.array(ac_cmap(0.5)[:3], dtype=np.float32)
        ac_u_rgb = np.repeat(base_rgb[None, :], z_valid.shape[0], axis=0)
        ac_w_rgb = np.repeat(base_rgb[None, :], z_valid.shape[0], axis=0)
        ac_vmin, ac_vmax = 0.0, 1.0
    else:
        cons_u_per_point = cons_u_grid[ix, iy]
        cons_w_per_point = cons_w_grid[ix, iy]

        cons_u_log = np.log1p(np.maximum(cons_u_per_point, 0.0))
        cons_w_log = np.log1p(np.maximum(cons_w_per_point, 0.0))

        if args.ac_mode == "linear":
            ac_vals = np.concatenate(
                [
                    np.log1p(np.maximum(cons_u_grid, 0.0))[np.isfinite(cons_u_grid)],
                    np.log1p(np.maximum(cons_w_grid, 0.0))[np.isfinite(cons_w_grid)],
                ]
            )
            if ac_vals.size:
                ac_vmin, ac_vmax = np.percentile(ac_vals, [5, 95])
            else:
                ac_vmin, ac_vmax = 0.0, 1.0
            ac_u_norm = _normalize(cons_u_log, ac_vmin, ac_vmax)
            ac_w_norm = _normalize(cons_w_log, ac_vmin, ac_vmax)
        elif args.ac_mode == "tanh":
            ac_scale = max(args.ac_scale, 1e-6)
            ac_u_norm = 0.5 * (np.tanh(cons_u_log / ac_scale) + 1.0)
            ac_w_norm = 0.5 * (np.tanh(cons_w_log / ac_scale) + 1.0)
            ac_vmin, ac_vmax = 0.0, 1.0
        elif args.ac_mode == "percentile_linear":
            ac_vals = np.concatenate(
                [
                    np.log1p(np.maximum(cons_u_grid, 0.0))[np.isfinite(cons_u_grid)],
                    np.log1p(np.maximum(cons_w_grid, 0.0))[np.isfinite(cons_w_grid)],
                ]
            )
            ref_sorted = np.sort(ac_vals) if ac_vals.size else np.array([0.0], dtype=np.float32)
            p_u = _percentile_rank(cons_u_log, ref_sorted)
            p_w = _percentile_rank(cons_w_log, ref_sorted)
            p_low, p_high = 0.05, 0.95
            p_u = np.clip(p_u, p_low, p_high)
            p_w = np.clip(p_w, p_low, p_high)
            ac_u_norm = (p_u - p_low) / (p_high - p_low)
            ac_w_norm = (p_w - p_low) / (p_high - p_low)
            ac_vmin, ac_vmax = 0.0, 1.0
        elif args.ac_mode == "percentile_tanh":
            ac_vals = np.concatenate(
                [
                    np.log1p(np.maximum(cons_u_grid, 0.0))[np.isfinite(cons_u_grid)],
                    np.log1p(np.maximum(cons_w_grid, 0.0))[np.isfinite(cons_w_grid)],
                ]
            )
            ref_sorted = np.sort(ac_vals) if ac_vals.size else np.array([0.0], dtype=np.float32)
            p_u = _percentile_rank(cons_u_log, ref_sorted)
            p_w = _percentile_rank(cons_w_log, ref_sorted)
            p_low, p_high = 0.05, 0.95
            p_u = np.clip(p_u, p_low, p_high)
            p_w = np.clip(p_w, p_low, p_high)
            p_u = (p_u - p_low) / (p_high - p_low)
            p_w = (p_w - p_low) / (p_high - p_low)
            t_u = (p_u - 0.5) * 2.0
            t_w = (p_w - 0.5) * 2.0
            ac_scale = max(args.ac_scale, 1e-6)
            ac_u_norm = 0.5 * (np.tanh(t_u / ac_scale) + 1.0)
            ac_w_norm = 0.5 * (np.tanh(t_w / ac_scale) + 1.0)
            ac_vmin, ac_vmax = 0.0, 1.0
        else:
            ac_vals = np.concatenate(
                [
                    np.log1p(np.maximum(cons_u_grid, 0.0))[np.isfinite(cons_u_grid)],
                    np.log1p(np.maximum(cons_w_grid, 0.0))[np.isfinite(cons_w_grid)],
                ]
            )
            if ac_vals.size:
                if args.ac_mode == "percentile":
                    ac_vmin, ac_vmax = np.percentile(ac_vals, [5, 95])
                else:
                    ac_vmin, ac_vmax = np.min(ac_vals), np.max(ac_vals)
            else:
                ac_vmin, ac_vmax = 0.0, 1.0
            ac_u_norm = _normalize(cons_u_log, ac_vmin, ac_vmax)
            ac_w_norm = _normalize(cons_w_log, ac_vmin, ac_vmax)

        ac_u_rgb = ac_cmap(ac_u_norm)[:, :3]
        ac_w_rgb = ac_cmap(ac_w_norm)[:, :3]

    value_scale = float(args.value_scale)
    ac_u_rgb = _apply_saturation(ac_u_rgb, sat_unw, value_scale)
    ac_w_rgb = _apply_saturation(ac_w_rgb, sat_w, value_scale)
    # One figure: 1 row x 2 columns (unweighted, weighted)
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), sharex=True, sharey=True)
    _scatter_points(
        axes[0],
        z_valid[:, 0],
        z_valid[:, 1],
        size_unw,
        ac_u_rgb,
        ac_cmap,
        ac_vmin,
        ac_vmax,
        blend_alpha,
    )
    _scatter_points(
        axes[1],
        z_valid[:, 0],
        z_valid[:, 1],
        size_w,
        ac_w_rgb,
        ac_cmap,
        ac_vmin,
        ac_vmax,
        blend_alpha,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
