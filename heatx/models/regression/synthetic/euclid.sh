#!/bin/bash

set -ex

python solve.py \
    --workdir ./tmp/default/heat/euclid \
    --config ./configs/euclidean.py \
    --model_workdir ./tmp/default \
    --model_config ./configs/default.py "$@"
