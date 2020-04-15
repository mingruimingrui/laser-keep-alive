#!/bin/sh
set -e

# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted directly from
# https://github.com/facebookresearch/LASER/blob/master/install_models.sh

s3="https://dl.fbaipublicfiles.com/laser/models"
filenames=(
    "bilstm.eparl21.2018-11-19.pt"
    "eparl21.fcodes" "eparl21.fvocab"
    "bilstm.93langs.2018-12-26.pt"
    "93langs.fcodes" "93langs.fvocab"
)

mkdir -p models
for filename in ${filenames[@]}; do
    if [ ! -f models/$filename ]; then
        echo "Downloading $filename"
        curl -o models/$filename $s3/$filename
    fi
done
echo "Download complete"
