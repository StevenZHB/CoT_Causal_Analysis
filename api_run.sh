#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 gpt-3.5-turbo Addition:6 cot0shot"
    echo "    bash $0 gpt-3.5-turbo Product:3 cot0shot"
    echo "    bash $0 gpt-3.5-turbo Addition:6 cot0shot 10"
    exit
fi

# run command
model_name=$1 # gpt-3.5-turbo, gpt-4, etc.
dataset=$2  # Addition, Product, ProofWriter, etc.
prompts=$3  # cot0shot, direct
nsamples=$4

if [ $# == '3' ]; then
    nsamples=-1
    if [[ "$dataset" == *"ProofWriter"* ]] || [ "$dataset" == "LOGIQA" ] || [[ "$dataset" == "FOLIO"* ]]; then
        echo "Dataset is $dataset. Ignore the nsamples argument."
        nsamples=600
    fi
fi

outdir=exp_cot/output
mkdir -p $outdir

echo `date`, Evaluating ${nsamples} samples from ${dataset} using prompts ${prompts} ...
python scripts/api_run.py --dataset $dataset --nsamples $nsamples --prompts $prompts --outdir $outdir \
                          --api_key $api_key --model_name $model_name