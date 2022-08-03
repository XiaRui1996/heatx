#!/bin/bash

set -ex

python solve.py \
    --manifold \
    --workdir ./tmp/default/heat/immersion \
    --config ./configs/immersion.py \
    --model_workdir ./tmp/default \
    --model_config ./configs/default.py "$@"
