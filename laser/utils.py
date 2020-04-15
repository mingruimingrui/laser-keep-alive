# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

"""Misc helper functions"""

import torch


def open_text_file(filepath: str, mode: str = 'r'):
    return open(filepath, mode=mode, encoding='utf-8', newline='\n')


def determine_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')
