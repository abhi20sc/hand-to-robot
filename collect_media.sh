#!/bin/bash
# Copies the figures the README uses from the results folders into media/.
set -e
mkdir -p media
cp ~/results/eval/success_heatmap.png                        media/heatmap_first.png
cp ~/results_chunking/wide/eval_exec5/success_heatmap.png    media/heatmap_final.png
cp ~/hand_demos_out/plots/Normal_10.png                      media/aperture_example.png
cp "$(ls ~/results_chunking/wide/eval_exec5/policy_success_*.gif | head -1)" media/policy_final.gif
cp ~/results/replay/gifs/Normal_10_side_by_side.gif          media/side_by_side.gif
ls -lh media
