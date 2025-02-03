#!/bin/bash

pop="$1"
task="$2"
method="$3"
eval="$4"

START=$(pwd)
OUT="$(mktemp -d /tmp/merge.XXXXX)"
cp "../bids_dataset/derivatives/rsa/sub-average/figures/pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_model-"*.png $OUT

cd "$OUT"

to_merge=""

for model in "symbolic" "IT" "both"; do
  convert "pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_model-${model}.png" "pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_model-empty.png" \
    -alpha off +repage \( -clone 0 -clone 1 -compose difference -composite -threshold 0 \) -delete 1 -alpha off -compose copy_opacity -composite tmp.png
  #convert tmp.png -alpha set -background none -channel A -evaluate multiply 0.8 +channel "${model}_alone.png"
  cp tmp.png "${model}_alone.png"
  to_merge="$to_merge ${model}_alone.png"
done

convert "pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_model-empty.png" $to_merge -background none -layers flatten +repage "pre_closer.png"

convert "pre_closer.png" -crop 10%x100% +repage image_%d.png
for f in image_*; do
  convert "$f" -fuzz 1% -trim +repage "tr_$f"
done

montage tr_image_* -geometry +5+0 -tile 10x1 tmp.png
convert tmp.png -transparent white output.png

cp output.png "pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_merged.png"

cp "pop-${pop}_task-${task}_method-${method}_eval-${eval}_slice_merged.png" "${START}/../bids_dataset/derivatives/rsa/sub-average/figures_stiched/"

rm -R "$OUT"
