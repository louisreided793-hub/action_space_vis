#!/bin/bash

python plot_weighted_distribution_shift.py \
  --precomputed canph_precomputed.npz \
  --weights canph_weights.json  --base-weight 3.5 \
  --out-fig canph_distribution_shift.png \
  --ac-mode percentile_tanh --clip \
  --base-dot-size 40  --weight-scale 5  --value-scale 1.0  --ac-scale 0.5 \
  --ac-colors "#ff0202ff, #d7a900ff, #00cc55ff"

python plot_weighted_distribution_shift.py \
  --precomputed canmh_precomputed.npz \
  --weights canmh_weights.json  --base-weight 3 \
  --out-fig canmh_distribution_shift.png \
  --ac-mode percentile_tanh --clip \
  --base-dot-size 40  --weight-scale 5  --value-scale 1.0  --ac-scale 0.5 \
  --ac-colors "#ff0202ff, #d7a900ff, #00cc55ff"

python plot_weighted_distribution_shift.py \
  --precomputed squareph_precomputed.npz \
  --weights squareph_weights.json  --base-weight 3.5 \
  --out-fig squareph_distribution_shift.png \
  --ac-mode percentile_tanh --clip \
  --base-dot-size 40  --weight-scale 5  --value-scale 1.0  --ac-scale 0.5 \
  --ac-colors "#ff0202ff, #d7a900ff, #00cc55ff"

python plot_weighted_distribution_shift.py \
  --precomputed squaremh_precomputed.npz \
  --weights squaremh_weights.json  --base-weight 3 \
  --out-fig squaremh_distribution_shift.png \
  --ac-mode percentile_tanh --clip \
  --base-dot-size 40  --weight-scale 5  --value-scale 1.0  --ac-scale 0.5 \
  --ac-colors "#ff0202ff, #d7a900ff, #00cc55ff"