#!/bin/bash

iskids="$1"

START=$(pwd)
OUT="$(mktemp -d /tmp/stich.XXXXX)"
cp ../bids_dataset/derivatives/bootstrap_clusters/figures_ventral/${iskids}_task-category_ctr-*_maths.png $OUT
cp background_3.png $OUT

cd "$OUT"

to_stich=""
for c in "c_number" "all_shapes"; do
  convert "${iskids}_task-category_ctr-${c}_maths.png" background_3.png -alpha off +repage \( -clone 0 -clone 1 -compose difference -composite -threshold 0 \) -delete 1 -alpha off -compose copy_opacity -composite tmp.png
  convert tmp.png -alpha set -background none -channel A -evaluate multiply 0.65 +channel "${c}_alone.png"
  to_stich="$to_stich ${c}_alone.png"
done

convert background_3.png $to_stich -gravity center -background None -layers Flatten tmp.png

convert tmp.png -transparent black  ${iskids}_maths_merged.png

cp "${iskids}_maths_merged.png" "${START}/../bids_dataset/derivatives/bootstrap_clusters/figures_ventral/"

rm -R "$OUT"
