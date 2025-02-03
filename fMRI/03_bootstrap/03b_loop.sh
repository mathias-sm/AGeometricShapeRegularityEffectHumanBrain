#!/bin/bash

for contrast in "motor" "ffa" "vwfa" "shape1" "shape3" "all_shapes"; do
  python 03b_inflated_plots.py --task=category --contrast=$contrast &
  python 03b_inflated_plots.py --task=category --contrast=$contrast --kids &
done

for contrast in "geom_theory" "geom_behavior_online" "geom_behavior_scanner"; do
  python 03b_inflated_plots.py --task=geometry --contrast=$contrast &
  python 03b_inflated_plots.py --task=geometry --contrast=$contrast --kids &
  python 03b_inflated_plots.py --task=geometryHard --contrast=$contrast &
done

wait

echo "DONE"
