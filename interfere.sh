#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 gpt-3.5-turbo Addition:6 cot0shot goldreason"
    echo "    bash $0 gpt-3.5-turbo Product:3 cot0shot randomreason"
    echo "    bash $0 gpt-3.5-turbo Addition:6 cot0shot defaultreason strongbias"
    echo "    bash $0 gpt-3.5-turbo Addition:6 cot0shot goldreason strongbias randomrole"
    exit
fi

# run command
model_name=$1 # gpt-3.5-turbo, gpt-4, etc.
dataset=$2  # Addition, Product, ProofWriter, etc.
prompts=$3  # cot0shot, direct
do_reason=$4 # defaultreason, goldreason, randomreason
do_bias=$5  # nobias, weakbias, strongbias
do_role=$6  # defaultrole, randomrole

if [ $# == '4' ]; then
    do_bias="nobias"
    do_role="defaultrole"  # default role
fi

if [ $# == '5' ]; then
    do_role="defaultrole"  # default role
fi

outdir=exp_cot/output

mkdir -p $outdir


echo `date`, Evaluating samples from ${dataset} using prompts ${prompts} with ${do_reason} and ${do_role} and ${do_bias} and seed ${SEED}...
python scripts/interfere.py  --dataset $dataset --prompt $prompts --outdir $outdir \
                             --do_reason $do_reason  --do_role $do_role --do_bias $do_bias \
                             --api_key $api_key --model_name $model_name