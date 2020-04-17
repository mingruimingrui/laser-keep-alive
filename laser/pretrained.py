# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download a pretrained model distributed by
the authors of the LASER project"""

import os
import subprocess
from typing import Tuple


S3_URL = 'https://dl.fbaipublicfiles.com/laser/models'
PRETRAINED_CODES_FILE = '93langs.fcodes'
PRETRAINED_MODEL_FILE = 'bilstm.93langs.2018-12-26.pt'
CACHE_DIR = os.path.join(os.environ['HOME'], '.cache/laser')


def download_file(url: str, filepath: str, force: bool = False):
    if not os.path.isfile(filepath) or force:
        assert subprocess.call('command -v curl', shell=True) == 0, \
            '`curl` is needed to download pretrained models'

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cmd = 'curl {} -o {}'.format(url, filepath)
        assert subprocess.call(cmd, shell=True) == 0, \
            'Failed to download from {}'.format(url)


def get_pretrained_model_paths(cache_dir: str = CACHE_DIR) -> Tuple[str, str]:
    """Download and cache the pretrained models and return the model save
    paths. On future calls, downloading will be skipped.

    Keyword Arguments:
        cache_dir {str} -- Directory to cache models (default: {CACHE_DIR})

    Returns:
        Tuple[str, str] -- (bpe_codes file, encoder model file)
    """
    for filename in [PRETRAINED_CODES_FILE, PRETRAINED_MODEL_FILE]:
        url = '{}/{}'.format(S3_URL, filename)
        filepath = os.path.join(CACHE_DIR, 'models', filename)
        download_file(url=url, filepath=filepath)

    return (
        os.path.join(CACHE_DIR, 'models', PRETRAINED_CODES_FILE),
        os.path.join(CACHE_DIR, 'models', PRETRAINED_MODEL_FILE)
    )
