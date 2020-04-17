# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

"""Encode a text file line by line"""

import os
import io
import argparse
from typing import Generator, List, Tuple

import threading

from time import time
from tqdm import tqdm

import torch
import numpy as np

from laser.data import Batcher, Batch
from laser.encoder import load_encoder_from_file
from laser.utils import open_text_file, determine_device
from laser import generator_utils as gen_utils
from laser.pretrained import CACHE_DIR, get_pretrained_model_paths

CACHE = {}


def add_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds options for this script to an existing parser
    options will be added inplace so there isn't a need to use the parser
    returned by this function.

    Arguments:
        parser {argparse.ArgumentParser} -- A given parser

    Returns:
        argparse.ArgumentParser -- The input parser
    """
    # I/O options
    io_group = parser.add_argument_group('I/O options')
    io_group.add_argument(
        'files', metavar='FILE', nargs='+', type=str,
        help='Text files to encode')
    io_group.add_argument(
        '-o', '--output', metavar='PREF', type=str, required=True,
        help='File to store outputs')
    io_group.add_argument(
        '--output-format', metavar='FORMAT', type=str,
        choices=['npy', 'csv'], default='npy',
        help='The output format')
    io_group.add_argument(
        '--output-fp16', action='store_true',
        help='Output results as fp16')

    # Model options
    model_group = parser.add_argument_group('Model options')
    model_group.add_argument(
        '-l', '--lang', metavar='LANG', type=str, required=True,
        help='Langauge identifier')
    model_group.add_argument(
        '-c', '--cache-dir', metavar='DIR', type=str, default=CACHE_DIR,
        help='Path to the directory where model is cached')
    model_group.add_argument(
        '-b', '--bpe-codes', metavar='FILE', type=str,
        help='Path to the bpe codes')
    model_group.add_argument(
        '-m', '--model', metavar='FILE', type=str,
        help='Path to the LASER bilstm model')

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


def download_pretrained_models(args: argparse.Namespace):
    if args.bpe_codes is None:
        args.bpe_codes = get_pretrained_model_paths(args.cache_dir)[0]
    if args.model is None:
        args.model = get_pretrained_model_paths(args.cache_dir)[1]


def create_batcher(dictionary: dict, args: argparse.Namespace) -> Batcher:
    return Batcher(
        lang=args.lang,
        bpe_codes=args.bpe_codes,
        dictionary=dictionary,
        max_seq_length=args.max_seq_len,
        max_sents=args.max_sents,
        max_tokens=args.max_tokens,
    )


def load_batcher(dictionary: dict, args: argparse.Namespace) -> Batcher:
    """Loads batcher into CACHE and on subsequent calls, retrieves batcher
    from cache.

    Arguments:
        dictionary {dict} -- Batcher dictionary
        args {argparse.Namespace} -- Parsed commandline options

    Returns:
        Batcher -- Text batcher
    """
    if 'batcher' not in CACHE:
        CACHE['batcher'] = create_batcher(dictionary, args)
    return CACHE['batcher']


def create_input_stream(
    args: argparse.Namespace
) -> Generator[str, None, None]:
    """Create a generator that loads lines from text files

    Arguments:
        args {argparse.Namespace} -- Parsed commandline options

    Yields:
        str -- Lines from input files
    """
    for filepath in args.files:
        with open_text_file(filepath, 'r') as f:
            for line in f:
                yield line


def create_batches(
    params: Tuple[
        List[str],
        dict,
        argparse.Namespace
    ]
) -> List[Batch]:
    """Worker function used to form batches from list of text

    Arguments:
        params {Tuple[List[str], dict, argparse.Namespace]} --
            Consists of the following
            - List of raw texts
            - Token dictionary
            - Parsed commandline options

    Returns:
        List[BATCH_TYPE] -- List of batches formed from raw texts.
    """
    chunk, dictionary, args = params
    batcher = load_batcher(dictionary, args)
    batches = list(batcher.make_batches(chunk))
    return batches


def append_results_to_file(
    embeddings: np.array,
    f: io.BufferedWriter,
    args: argparse.Namespace
):
    """Appends embeddings to output file

    Arguments:
        embeddings {np.array} -- Newly generated embeddings
        f {io.BufferedWriter} -- A bytes writier to store outputs
        args {argparse.Namespace} -- Parsed commandline options

    Raises:
        ValueError: output_format is invalid
    """
    output_dtype = np.float16 if args.output_fp16 else np.float32
    if embeddings.dtype != output_dtype:
        embeddings = embeddings.astype(output_dtype)

    if args.output_format == 'npy':
        embeddings.tofile(f)

    elif args.output_format == 'csv':
        import pandas as pd
        df = pd.DataFrame(embeddings)
        buffer = df.to_csv(header=False, index=False)
        assert buffer.endswith('\n')
        f.write(buffer.encode('utf-8'))

    else:
        msg = 'Unexpected output format "{}"'.format(args.output_format)
        raise ValueError(msg)


def main(args):
    if args.num_workers >= 1:
        import multiprocessing
        num_workers = args.num_workers
    else:
        from multiprocessing import dummy as multiprocessing
        num_workers = 1

    # Download pretrained model if needed
    download_pretrained_models(args)

    start_time = time()

    # Load encoder
    device = determine_device(args.gpu)
    encoder, dictionary = load_encoder_from_file(args.model)
    encoder = encoder.eval()
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
    def process_batches(batches: List[Batch]) -> Tuple[List[str], np.ndarray]:
        chunk_size = sum([len(batch[0]) for batch in batches])
        chunk_texts = [None] * chunk_size
        chunk_embeddings = [None] * chunk_size
        for b in batches:
            outputs = encoder.forward(
                b.tokens.to(device),
                b.lengths.to(device),
            ).sentemb
            outputs = outputs.detach().cpu().numpy()

            for i, text, embedding in zip(b.indices, b.texts, outputs):
                chunk_texts[i] = text
                chunk_embeddings[i] = embedding

        return chunk_texts, np.array(chunk_embeddings)

    # Remove existing output file if needed
    output_path = os.path.abspath(args.output)
    if os.path.isfile(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Loop through each chunk
    # 1. Create batches (on subprocesses)
    # 2. Inference with bilstm
    # 3. Save results to file
    with open(output_path, 'wb') as f:
        with tqdm(unit='sent') as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                t0 = time()
                for batches in pool.imap(create_batches, create_loader()):
                    semaphore.release()

                    t1 = time()
                    _, embeddings = process_batches(batches)

                    t2 = time()
                    torch.cuda.empty_cache()
                    append_results_to_file(embeddings, f, args)

                    t3 = time()
                    pbar.set_postfix({
                        'data_time': t1 - t0,
                        'inference_time': t2 - t1,
                        'write_time': t3 - t2,
                    })
                    pbar.update(len(embeddings))
                    t0 = time()

    time_taken = time() - start_time
    print('Finished in {:.1f}s'.format(time_taken))
