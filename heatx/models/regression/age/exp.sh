#!/bin/bash

set -ex

python exp.py \
    --manifold score \
    --workdir ./tmp/utkface \
    --config ./configs/score.py \
    --manifold_workdir ../../generative/sde/tmp/vesde \
    --manifold_config ../../generative/sde/configs/vesde.py \
    "$@"
