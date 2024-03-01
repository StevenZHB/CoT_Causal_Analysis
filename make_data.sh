#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

datadir=./data

nsamples=500
python scripts/data_builder.py --dataset Addition --ndigits 6 --nsamples $nsamples --datadir $datadir
python scripts/data_builder.py --dataset Addition --ndigits 9 --nsamples $nsamples --datadir $datadir
python scripts/data_builder.py --dataset Product --ndigits 2 --nsamples $nsamples --datadir $datadir
python scripts/data_builder.py --dataset Product --ndigits 3 --nsamples $nsamples --datadir $datadir
