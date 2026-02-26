#!/bin/bash
set -euo pipefail

python plot_weighted_distribution_shift.py \
  --precomputed canph_precomputed.npz \
  --weights canph_weights.json \
  --base-weight 3.5 \
  --out-fig canph_distribution_shift.png \
  --grid 64 --error-quantile 0.55 --weight-scale 1.0 --mask-dilate 2

python plot_weighted_distribution_shift.py \
  --precomputed canmh_precomputed.npz \
  --weights canmh_weights.json \
  --base-weight 3 \
  --out-fig canmh_distribution_shift.png \
  --grid 64 --error-quantile 0.55 --weight-scale 1.0 --mask-dilate 2

python plot_weighted_distribution_shift.py \
  --precomputed squareph_precomputed.npz \
  --weights squareph_weights.json \
  --base-weight 3.5 \
  --out-fig squareph_distribution_shift.png \
  --grid 64 --error-quantile 0.55 --weight-scale 1.0 --mask-dilate 2

python plot_weighted_distribution_shift.py \
  --precomputed squaremh_precomputed.npz \
  --weights squaremh_weights.json \
  --base-weight 3 \
  --out-fig squaremh_distribution_shift.png \
  --grid 64 --error-quantile 0.55 --weight-scale 1.0 --mask-dilate 2
