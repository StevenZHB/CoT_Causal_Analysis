#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

outdir=exp_cot/output
model='gpt-3.5-turbo gpt-4'
datasets='Addition:6 Product:3'

# check the reasoning steps of arithmetic problems
for M in $model; do
  for D in $datasets; do
    python scripts/check.py --model_name $M --dataset $D
  done
done
