#!/bin/bash

for contrast in "motor" "ffa" "vwfa" "shape1" "shape3" "all_shapes" "house" "tool"; do
  python 03d_output_cluster_table.py --task=category --contrast=$contrast &
  python 03d_output_cluster_table.py --task=category --contrast=$contrast --kids &
done

for contrast in "geom_theory" "geom_behavior_online" "geom_behavior_scanner"; do
  python 03d_output_cluster_table.py --task=geometry --contrast=$contrast &
  python 03d_output_cluster_table.py --task=geometry --contrast=$contrast --kids &
  python 03d_output_cluster_table.py --task=geometryHard --contrast=$contrast &
done

wait

echo "DONE"
