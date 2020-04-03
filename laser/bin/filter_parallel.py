"""Filter a parallel corpus
"""

import os
import argparse
from typing import Generator, List, Tuple

import threading

from time import time
from tqdm import tqdm

import torch
import numpy as np

from laser.data import Batcher, BATCH_TYPE
from laser.encoder import load_model_from_file
from laser.utils import open_text_file
from laser import generator_utils as gen_utils

CACHE = {}


def add_options(parser: argparse.ArgumentParser):
    # I/O options
    io_group = parser.add_argument_group('I/O options')
    io_group.add_argument(
        'prefixes', metavar='PREFS', nargs='+', type=str,
        help='Parallel corpus file prefixes')
    io_group.add_argument(
        '-o', '--output', metavar='PREF', type=str, required=True,
        help='Prefix to a file to store outputs')

    # Model options
    model_group = parser.add_argument_group('Model options')
    model_group.add_argument(
        '-l', '--langs', metavar='LANG', type=str, nargs=2, required=True,
        help='Source and target langauge identifier')
    model_group.add_argument(
        '-b', '--bpe-path', metavar='FILE', type=str, nargs=2, required=True,
        help='Path to the bpe code and vocab files')
    model_group.add_argument(
        '-m', '--model', metavar='FILE', type=str, required=True,
        help='Path to the LASER bilstm model')

    # Filter options
    filter_group = parser.add_argument_group('Filter options')
    filter_group.add_argument(
        '--threshold', metavar='FLOAT', type=float, default=1.04,
        help='L2 distance threshold to filter sentence pairs')
    filter_group.add_argument(
        '--output-filtered', action='store_true',
        help='Write filtered data to <output_file>.removed')

    # Inference options
    inference_group = parser.add_argument_group('Inference options')
    inference_group.add_argument(
        '--max-seq-len', metavar='INT', type=int, default=256,
        help='Maximum number of tokens per sentence')
    inference_group.add_argument(
        '--max-sents', metavar='INT', type=int,
        help='Maximum number sentences per batch')
    inference_group.add_argument(
        '--max-tokens', metavar='INT', type=int, default=12000,
        help='Maximum number tokens per batch')
    inference_group.add_argument(
        '--gpu', action='store_true',
        help='Run model with CPU?')
    inference_group.add_argument(
        '--fp16', action='store_true',
        help='Run model with FP16?')

    # Data loading options
    data_group = parser.add_argument_group('Data loading options')
    data_group.add_argument(
        '--chunk-size', metavar='INT', type=int, default=10000,
        help='The chunk size that input files should be read in')
    data_group.add_argument(
        '--num-workers', metavar='INT', type=int, default=4,
        help='Number of workers to use to generate new batches')

    return parser


def determine_device(args: argparse.Namespace):
    if args.gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def create_batchers(
    dictionary: dict,
    args: argparse.Namespace,
) -> Tuple[Batcher, Batcher]:
    kwargs = dict(
        bpe_codes=args.bpe_path[0],
        bpe_vocab=args.bpe_path[1],
        dictionary=dictionary,
        max_seq_length=args.max_seq_len,
        max_sents=args.max_sents,
        max_tokens=args.max_tokens,
    )
    src_batcher = Batcher(lang=args.langs[0], **kwargs)
    tgt_batcher = Batcher(lang=args.langs[1], **kwargs)
    return src_batcher, tgt_batcher


def create_input_stream(args: argparse.Namespace):
    """Reads all files referenced by prefixes and langs

    Arguments:
        args {argparse.Namespace} -- Parsed command line options

    Yields:
        Tuple[str, str] -- src_line, tgt_line
    """
    for pref in args.prefixes:
        src_filepath = '{}.{}'.format(pref, args.langs[0])
        tgt_filepath = '{}.{}'.format(pref, args.langs[1])

        with open_text_file(src_filepath, 'r') as fsrc:
            with open_text_file(tgt_filepath, 'r') as ftgt:
                for src_line, tgt_line in zip(fsrc, ftgt):
                    yield src_line, tgt_line


def create_batches(
    params: Tuple[List[Tuple[str, str]], dict, argparse.Namespace]
) -> Tuple[List[BATCH_TYPE], List[BATCH_TYPE]]:
    chunk, dictionary, args = params
    src_batcher, tgt_batcher = load_batchers(dictionary, args)
    src_lines, tgt_lines = zip(*chunk)
    src_batches = list(src_batcher.make_batches(src_lines))
    tgt_batches = list(tgt_batcher.make_batches(tgt_lines))
    return src_batches, tgt_batches


def load_batchers(
    dictionary: dict,
    args: argparse.Namespace,
) -> Tuple[Batcher, Batcher]:
    if 'src_batcher' not in CACHE:
        CACHE['src_batcher'], CACHE['tgt_batcher'] = \
            create_batchers(dictionary, args)
    return CACHE['src_batcher'], CACHE['tgt_batcher']


def init_output_files(args: argparse.Namespace):
    output_pref = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_pref), exist_ok=True)
    with open_text_file('{}.{}'.format(output_pref, args.langs[0]), 'w') as f:
        pass
    with open_text_file('{}.{}'.format(output_pref, args.langs[1]), 'w') as f:
        pass


def append_results_to_file(
    src_texts: List[str],
    src_embeddings: np.ndarray,
    tgt_texts: List[str],
    tgt_embeddings: np.ndarray,
    args: argparse.Namespace,
):
    assert len(src_texts) == len(src_embeddings)
    assert len(tgt_texts) == len(tgt_embeddings)
    assert len(src_texts) == len(tgt_texts)

    src_outpath = '{}.{}'.format(args.output, args.langs[0])
    tgt_outpath = '{}.{}'.format(args.output, args.langs[1])
    l2_dists = np.sqrt(np.sum((src_embeddings - tgt_embeddings) ** 2, axis=1))

    with open_text_file(src_outpath, 'a') as fsrc:
        with open_text_file(tgt_outpath, 'a') as ftgt:
            for i, d in enumerate(l2_dists):
                if d <= args.threshold:
                    fsrc.write(src_texts[i])
                    ftgt.write(tgt_texts[i])


def main(args):
    if args.num_workers >= 1:
        import multiprocessing
        num_workers = args.num_workers
    else:
        from multiprocessing import dummy as multiprocessing
        num_workers = 1

    start_time = time()

    # Load encoder
    device = determine_device(args)
    encoder, dictionary = load_model_from_file(args.model)
    encoder = encoder.to(device)
    if args.fp16:
        encoder = encoder.half()

    # Define create_loader fn
    semaphore = threading.Semaphore(num_workers)

    def create_loader():
        for chunk in gen_utils.chunk(
            create_input_stream(args),
            size=args.chunk_size,
        ):
            semaphore.acquire()
            yield chunk, dictionary, args

    # Define process_batches fn
    @torch.no_grad()
    def process_batches(batches):
        chunk_size = sum([len(batch[0]) for batch in batches])
        chunk_texts = [None] * chunk_size
        chunk_embeddings = [None] * chunk_size
        for indices, texts, tokens, lengths in batches:
            outputs = encoder.forward(
                tokens.to(device),
                lengths.to(device)
            )['sentemb']
            outputs = outputs.detach().cpu().numpy()

            for i, text, embedding in zip(indices, texts, outputs):
                chunk_texts[i] = text
                chunk_embeddings[i] = embedding

        return chunk_texts, np.array(chunk_embeddings)

    # Loop through each chunk
    # 1. Create batches (on subprocesses)
    # 2. Inference with bilstm
    # 3. Save results to file
    with tqdm() as pbar:
        with

    with multiprocessing.Pool(args.num_workers) as pool:
        for src_batches, tgt_batches in pool.imap(worker_fn, create_loader()):
            semaphore.release()
            src_texts, src_embeddings = process_chunk(src_batches)
            tgt_texts, tgt_embeddings = process_chunk(tgt_batches)