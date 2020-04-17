# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data processing and loading classes and helper functions"""

import os
import warnings
from typing import NamedTuple, Iterator, Generator, List, Optional

import torch
import numpy as np


MOSES_CHUNK_SIZE = 128


class Tokenizer(object):

    def __init__(self, lang: str = 'en'):
        """Create a tokenizer instance for a specific language.
        Take note that a caveat is that this tokenizer is not picklable.

        Does the following in sequence
        1. Punct normalize
        2. Lang specific moses tokenize
        3. Lang specific addtional segmentation

        Keyword Arguments:
            lang {str} -- Language identifier (default: {'en'})

        Raises:
            ImportError: Requires sacremoses
        """
        try:
            from mosestokenizer import \
                MosesPunctuationNormalizer, MosesTokenizer
        except ImportError:
            raise ImportError('Please install sacremoses')

        self.lang = lang
        self.punct_normalizer = MosesPunctuationNormalizer(lang=lang)
        self.tokenizer = MosesTokenizer(lang=lang)

        if self.lang == 'zh':
            import opencc
            self.opencc_converter = opencc.OpenCC('tw2s')

            import jieba
            self.jieba_tokenizer = jieba.dt

        elif self.lang == 'jp':
            import MeCab
            self.wakati = MeCab.Tagger('-Owakati')

        elif self.lang == 'el':
            import transliterate
            self.translit = transliterate.translit

    def tokenize(self, texts: List[str]) -> List[str]:
        """Tokenizes a string. Output is string rather than list of tokens.

        Arguments:
            text {List[str]} -- Raw input texts

        Returns:
            List[str] -- Tokenized sentences
        """
        texts = [' '.join(t.lower().split()) for t in texts]
        tokenized_texts = []
        for i in range(0, len(texts), MOSES_CHUNK_SIZE):
            chunk = texts[i:(i + MOSES_CHUNK_SIZE)]

            # Normalize punctuations
            [self.punct_normalizer.writeline(t) for t in chunk]

            # Tokenize
            [self.tokenizer.writeline(
                self.punct_normalizer.readline()
            ) for _ in chunk]

            tokenized_texts += [self.tokenizer.readline() for _ in chunk]
            # This may seem like a lot of for loops
            # but the overhead of tokenization should be much higher
            # Also this is an efficient way to buffer I/O to/from the
            # perl scripts

            # Random note: it might be possible to run these piped processes
            # on a separate thread.
            # Would not improve the current performance of this script by
            # much but still a cool experiment.

        if self.lang == 'zh':
            texts = [
                ' '.join(self.jieba_tokenizer.cut(
                    self.opencc_converter.convert(text),
                    cut_all=False, HMM=True
                ))
                for text in texts
            ]

        elif self.lang == 'jp':
            texts = [
                self.wakati.parse(text) if len(text) > 0 else ''
                for text in texts
            ]

        elif self.lang == 'el':
            texts = [
                self.translit(text, language_code='el')
                for text in texts
            ]

        return texts


class Batch(NamedTuple):
    indices: List[int]
    texts: List[str]
    tokens: torch.Tensor
    lengths: torch.Tensor


class Batcher(object):

    def __init__(
        self,
        lang: str,
        bpe_codes: str,
        dictionary: dict,
        max_seq_length: int = 256,
        max_sents: Optional[int] = None,
        max_tokens: Optional[int] = 12000
    ):
        self.lang = lang

        # Load tokenizer and bpe model
        try:
            from fastBPE import fastBPE
        except ImportError:
            raise ImportError('Please install fastBPE first')
        assert os.path.isfile(bpe_codes)

        self.tokenizer = Tokenizer(lang=lang)
        self.bpe_model = fastBPE(bpe_codes)

        # Set dictionary
        assert (
            '<pad>' in dictionary and
            '</s>' in dictionary and
            '<unk>' in dictionary
        ), 'dictionary appears to be invalid'

        self.dictionary = dictionary
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

        # Setup max_seq_length, max_sents and max_tokens
        if max_sents is None and max_tokens is None:
            warnings.warn(
                'Using batcher without specifying max_sents and max_tokens '
                'defaulting to using max_sents of 1. '
                'This can be highly inefficient.'
            )
            max_sents = 1

        self.max_seq_length = max_seq_length
        self.max_sents = max_sents
        self.max_tokens = max_tokens

    def _text_to_token_ids(self, text: str) -> List[int]:
        """Convert a text into a list of token ids
        Sentence that is too long will be truncated

        Arguments:
            text {str} -- BPE encoded text

        Returns:
            List[int] -- List of token ids
        """
        tokens = [self.dictionary.get(t, self.unk_index) for t in text.split()]
        if len(tokens) > self.max_seq_length - 1:
            warnings.warn((
                'Found sentence with over {} tokens '
                'sentence will be truncated to a valid length'
            ).format(self.max_seq_length))
            tokens = tokens[:(self.max_seq_length - 1)]
        tokens.append(self.eos_index)
        return tokens

    def _process_texts(self, texts: List[str]) -> List[List[int]]:
        """Process a list of texts

        Arguments:
            texts {List[str]} -- Raw texts

        Returns:
            List[List[int]] -- List of token ids
        """
        texts = self.tokenizer.tokenize(texts)
        texts = self.bpe_model.apply(texts)
        return [self._text_to_token_ids(text) for text in texts]

    def _collate_tokens(self, tokens: List[List[int]]) -> torch.Tensor:
        """Right pad a list of token ids

        Arguments:
            tokens {List[List[int]]} -- List of token ids

        Returns:
            torch.Tensor -- A 2D tensor of shape [batch_size, seq_len]
        """
        batch_length = max([len(t) for t in tokens])
        batch_length = min(batch_length, self.max_seq_length)

        def pad_toks(toks: List[int]) -> List[int]:
            length = len(toks)
            if length < batch_length:
                toks += [self.pad_index] * (batch_length - length)
            return toks

        return torch.LongTensor([pad_toks(t) for t in tokens])

    def make_batches(
        self, texts: Iterator[str]
    ) -> Generator[Batch, None, None]:
        """Create batches from raw texts
        Sequences in each batch will be sorted based on longest to shortest.
        Batches will also be right padded.

        TODO: Should warn if too many unknown tokens are produced

        Arguments:
            texts {Iterator[str]} -- A list of raw texts

        Yields:
            List[int] -- batch_indices
                The original indices of input text
            List[str] -- batch_texts
                The input texts for this batch
            torch.Tensor -- batch_tokens
                The token_ids padded into a 2D tensor of shape
                [batch_size, seq_len]
            torch.Tensor -- batch_lengths
                The unpadded sequence lengths of each entry.
                A 1D tensor of shape [batch_size]
        """
        tokens = self._process_texts(texts)
        lengths = [len(t) for t in tokens]
        indices = np.argsort(lengths)[::-1]

        batch_indices = []
        batch_texts = []
        batch_tokens = []
        max_batch_length = 0
        for i in indices:
            if (
                (
                    self.max_sents is not None and
                    len(batch_tokens) >= self.max_sents
                ) or (
                    self.max_tokens is not None and (
                        max(max_batch_length, lengths[i]) *
                        (len(batch_tokens) + 1) > self.max_tokens
                    )
                )
            ):
                # yield cur batch
                batch_tokens = self._collate_tokens(batch_tokens)
                yield Batch(
                    indices=batch_indices,
                    texts=batch_texts,
                    tokens=batch_tokens,
                    lengths=torch.LongTensor([len(t) for t in batch_tokens])
                )
                batch_indices = []
                batch_texts = []
                batch_tokens = []
                max_batch_length = 0

            batch_indices.append(i)
            batch_texts.append(texts[i])
            batch_tokens.append(tokens[i])
            max_batch_length = max(max_batch_length, lengths[i])

        if len(batch_tokens) > 0:
            batch_tokens = self._collate_tokens(batch_tokens)
            yield Batch(
                indices=batch_indices,
                texts=batch_texts,
                tokens=batch_tokens,
                lengths=torch.LongTensor([len(t) for t in batch_tokens])
            )
