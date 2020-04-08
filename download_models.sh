#!/bin/sh
set -e

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
        curl -o models/$filename $s3/$filename
    fi
done
