#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 Direct.vs.CoT"
    echo "    bash $0 GoldCoT.vs.Default"
    echo "    bash $0 RandCoT.vs.Default"
    echo "    bash $0 'RandRole.vs.Default|DefaultCoT'"
    echo "    bash $0 'RandRole.vs.Default|GoldCoT'"
    echo "    bash $0 'RandBias.vs.Default|DefaultCoT'"
    echo "    bash $0 'RandBias.vs.Default|GoldCoT'"
    exit
fi

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# run command
test=$1

outdir=exp_cot/output

if [ $test == 'Direct.vs.CoT' ]; then
  # Direct .vs. CoT
  model='llama2-7b-chat llama2-70b-chat gpt-3.5-turbo gpt-4'
  datasets='Addition:6 Addition:9 Product:2 Product:3 GSM8K ProofWriter FOLIO LOGIQA'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.direct.math_teacher.${M}.json
      B=$outdir/output.${D}.cot0shot.math_teacher.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'GoldCoT.vs.Default' ]; then
  # golden CoT .vs. default CoT, given constant Instruction
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.defaultreason.${M}.json
      B=$outdir/output.${D}.cot0shot.goldreason.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'RandCoT.vs.Default' ]; then
  # random CoT .vs. default CoT, given constant Instruction
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter FOLIO LOGIQA'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.defaultreason.${M}.json
      B=$outdir/output.${D}.cot0shot.randomreason.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'RandRole.vs.Default|DefaultCoT' ]; then
  # random role .vs. default role, given default CoT
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter FOLIO LOGIQA'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.defaultreason.${M}.json
      B=$outdir/output.${D}.cot0shot.defaultreason_randomrole.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'RandRole.vs.Default|GoldCoT' ]; then
  # random role .vs. default role, given default CoT
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter FOLIO LOGIQA'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.goldreason.${M}.json
      B=$outdir/output.${D}.cot0shot.goldreason_randomrole.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'RandBias.vs.Default|DefaultCoT' ]; then
  # random bias .vs. default, given default CoT
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter FOLIO LOGIQA'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.defaultreason.${M}.json
      B=$outdir/output.${D}.cot0shot.defaultreason_strongbias.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

elif [ $test == 'RandBias.vs.Default|GoldCoT' ]; then
  # random bias .vs. default, given golden CoT
  model='gpt-3.5-turbo'  #  gpt-4
  datasets='Addition:6 Product:3 GSM8K ProofWriter'
  for M in $model; do
    for D in $datasets; do
      A=$outdir/output.${D}.cot0shot.goldreason.${M}.json
      B=$outdir/output.${D}.cot0shot.goldreason_strongbias.${M}.json
      python scripts/mcnemar_test.py --group_a $A  --group_b $B
    done
  done

else
  echo `date`, Unknown test ${test}
fi
