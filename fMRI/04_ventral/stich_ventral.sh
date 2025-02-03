#!/bin/bash

iskids="$1"

START=$(pwd)
OUT="$(mktemp -d /tmp/stich.XXXXX)"
cp ../bids_dataset/derivatives/bootstrap_clusters/figures_ventral/${iskids}_task-category_ctr-*_ventral.png $OUT
cp background.png $OUT

cd "$OUT"

to_stich=""
for c in "vwfa" "ffa" "house" "tool"; do
  convert "${iskids}_task-category_ctr-${c}_ventral.png" background.png -alpha off +repage \( -clone 0 -clone 1 -compose difference -composite -threshold 0 \) -delete 1 -alpha off -compose copy_opacity -composite tmp.png
  convert tmp.png -alpha set -background none -channel A -evaluate multiply 0.8 +channel "${c}_alone.png"
  to_stich="$to_stich ${c}_alone.png"
done

convert background.png $to_stich -gravity center -background None -layers Flatten tmp.png

convert tmp.png -transparent black  ${iskids}_ventral_merged.png

cp "${iskids}_ventral_merged.png" "${START}/../bids_dataset/derivatives/bootstrap_clusters/figures_ventral/"

rm -R "$OUT"
