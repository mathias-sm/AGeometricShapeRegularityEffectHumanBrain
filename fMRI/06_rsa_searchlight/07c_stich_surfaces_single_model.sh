#!/bin/bash

pop="$1"
task="$2"
method="$3"
eval="$4"

START=$(pwd)
OUT="$(mktemp -d /tmp/stich.XXXXX)"

cp "../bids_dataset/derivatives/rsa/sub-average/figures/pop-${pop}_task-${task}_method-${method}_eval-${eval}_"*.png "$OUT"

cd "$OUT" || exit 1

for f in *.png; do convert "$f" -transparent white tmp.png; mv tmp.png "$f"; done

for model in "IT" "symbolic" "dino_last" "skel_1" "skel_2"; do

  convert -rotate -90 "pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-ventral_hemisphere-left_model-${model}_isolated.png" b1.png
  convert -rotate -90 "pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-ventral_hemisphere-right_model-${model}_isolated.png" b2.png

  convert -size 4000x4000 xc:transparent \
            -draw "image over 0000,0000 0,0 'pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-lateral_hemisphere-left_model-${model}_isolated.png'" \
            -draw "image over 2100,0000 0,0 'pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-lateral_hemisphere-right_model-${model}_isolated.png'" \
            -draw "image over 0840,0680 0,0 'b1.png'" \
            -draw "image over 1360,0680 0,0 'b2.png'" \
            -draw "image over 0000,0990 0,0 'pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-medial_hemisphere-left_model-${model}_isolated.png'" \
            -draw "image over 2120,0990 0,0 'pop-${pop}_task-${task}_method-${method}_eval-${eval}_view-medial_hemisphere-right_model-${model}_isolated.png'" \
            "output.png"

  convert output.png -fuzz 1% -trim +repage "pop-${pop}_task-${task}_method-${method}_eval-${eval}_model-${model}_merged_isolated.png" 

  rm b1.png b2.png output.png

  cp "pop-${pop}_task-${task}_method-${method}_eval-${eval}_model-${model}_merged_isolated.png" "${START}/../bids_dataset/derivatives/rsa/sub-average/figures_stiched/"

  echo "done with $1 $2 $3 $4 $model"

done

#rm -R "$OUT"
