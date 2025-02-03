#!/bin/bash

iskids="$1"
task="$2"
contrast="$3"
thresh="$4"

START=$(pwd)
OUT="$(mktemp -d /tmp/stich.XXXXX)"

cp "./figs_uncorr/${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}"*.png "$OUT"

cd "$OUT" || exit 1

for f in *.png; do convert "$f" -transparent white tmp.png; mv tmp.png "$f"; done

convert -rotate 90 "${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-ventral_hemisphere-left.png" b1.png
convert -rotate 90 "${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-ventral_hemisphere-right.png" b2.png

convert -size 4000x4000 xc:transparent \
          -draw "image over 0000,0000 0,0 '${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-lateral_hemisphere-left.png'" \
          -draw "image over 2100,0000 0,0 '${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-lateral_hemisphere-right.png'" \
          -draw "image over 0795,0630 0,0 'b1.png'" \
          -draw "image over 1315,0630 0,0 'b2.png'" \
          -draw "image over 0000,0990 0,0 '${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-medial_hemisphere-left.png'" \
          -draw "image over 2120,0990 0,0 '${iskids}_plot-surfInf_second-level_thresh-${thresh}_task-${task}_contrast-${contrast}_view-medial_hemisphere-right.png'" \
          "output.png"


convert output.png -fuzz 1% -trim +repage "${iskids}_plot-surfInf_thresh-${thresh}_task-${task}_contrast-${contrast}_merged.png" 

rm b1.png b2.png output.png

echo "${iskids}_plot-surfInf_thresh-${thresh}_task-${task}_contrast-${contrast}_merged.png"
cp "${iskids}_plot-surfInf_thresh-${thresh}_task-${task}_contrast-${contrast}_merged.png" "${START}/figs_uncorr_stiched/"

rm -R "$OUT"
