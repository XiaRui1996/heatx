#!/bin/bash

set -ex

python solve.py \
    --workdir ./tmp/utkface/heat/score \
    --config ./configs/score.py \
    --model_workdir ./tmp/utkface \
    --model_config ./configs/default.py \
    --manifold_workdir ../../generative/sde/tmp/utkface \
    --manifold_config ../../generative/sde/configs/utkface.py
