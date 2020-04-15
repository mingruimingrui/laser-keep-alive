# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from laser import bin as laser_bin


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m laser',
        description='Language-Agnostic SEntence Representations',
    )
    subparsers = parser.add_subparsers(dest='subcommand')

    encode_parser = subparsers.add_parser(
        'encode', help='Encode a text file line by line')
    laser_bin.encode_texts.add_options(encode_parser)

    filter_parser = subparsers.add_parser(
        'filter', help='Filter a parallel corpus based on similarity')
    laser_bin.filter_parallel.add_options(filter_parser)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.subcommand == 'encode':
        laser_bin.encode_texts.main(args)

    elif args.subcommand == 'filter':
        laser_bin.filter_parallel.main(args)

    else:
        parser.print_help()
