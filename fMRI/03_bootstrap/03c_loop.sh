#!/bin/bash

for contrast in "motor" "ffa" "vwfa" "shape1" "shape3" "all_shapes"; do
  ./03c_stich_surfaces.sh adults category $contrast &
  ./03c_stich_surfaces.sh kids category $contrast &
done

for contrast in "geom_theory" "geom_behavior_online" "geom_behavior_scanner"; do
  ./03c_stich_surfaces.sh adults geometry $contrast &
  ./03c_stich_surfaces.sh kids geometry $contrast &
  ./03c_stich_surfaces.sh adults geometryHard $contrast &
done

wait

echo "DONE"
