import argparse
from laser import bin as laser_bin


def make_parser() -> argparse.ArgumentParser:
    description = (
        'Language-Agnostic SEntence Representations\n'
        'https://github.com/facebookresearch/LASER/'
    )

    parser = argparse.ArgumentParser(description=description)
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
