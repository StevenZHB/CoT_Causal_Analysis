#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 gpt-3.5-turbo ProofWriter cot0shot"
    exit
fi

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# run command
model_name=$1  # gpt-3.5-turbo
dataset=$2  # Addition, Product, ProofWriter, etc.
prompts=$3  # cot0shot, direct


outdir=exp_cot/random_reason
mkdir -p $outdir

echo `date`, Making random reasons from ${dataset} using prompts ${prompts} and model ${model_name}...
python scripts/random_reason.py --dataset $dataset --prompt $prompts --model_name $model_name --api_key $api_key