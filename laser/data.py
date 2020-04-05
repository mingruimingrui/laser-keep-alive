"""Data processing and loading classes and helper functions
"""

import warnings
import unicodedata
from typing import Iterator, Generator, Tuple, List, Optional

import torch
import numpy as np

BATCH_TYPE = Tuple[List[int], List[str], torch.Tensor, torch.Tensor]


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
            from sacremoses import MosesPunctNormalizer, MosesTokenizer
        except ImportError:
            raise ImportError('Please install sacremoses')

        self.lang = lang
        self.punct_normalizer = MosesPunctNormalizer(lang=lang)
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

    def tokenize(self, text: str) -> str:
        """Tokenizes a string. Output is string rather than list of tokens.

        Arguments:
            text {str} -- Raw input string

        Returns:
            str -- Tokens joined by whitespace
        """
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = self.punct_normalizer.normalize(text)
        text = self.tokenizer.tokenize(
            text,
            aggressive_dash_splits=False,
            return_str=True,
            escape=False
        )

        if self.lang == 'zh':
            text = self.opencc_converter.convert(text)
            tokens = self.jieba_tokenizer.cut(text, cut_all=False, HMM=True)
            text = ' '.join(tokens)

        elif self.lang == 'jp':
            text = self.wakati.parse(text)

        elif self.lang == 'el':
            text = self.translit(text, language_code='el')

        return text


class Batcher(object):

    def __init__(
        self,
        lang: str,
        bpe_codes: str,
        bpe_vocab: str,
        dictionary: dict,
        max_seq_length: int = 256,
        max_sents: Optional[int] = None,
        max_tokens: Optional[int] = None
    ):
        self.lang = lang

        # Load tokenizer and bpe model
        try:
            from fastBPE import fastBPE
        except ImportError:
            raise ImportError('Please install fastBPE first')

        self.tokenizer = Tokenizer(lang=lang)
        self.bpe_model = fastBPE(bpe_codes, bpe_vocab)

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

    def _process_texts(self, texts: List[str]) -> List[List[int]]:
        """Process a list of texts

        Arguments:
            texts {List[str]} -- Raw texts

        Returns:
            List[List[int]] -- List of token ids
        """
        texts = [self.tokenizer.tokenize(t) for t in texts]
        texts = self.bpe_model.apply(texts)
        return [
            [
                self.dictionary.get(t, self.unk_index)
                for t in text.split()
            ] + [self.eos_index] for text in texts
        ]

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
            if length > self.max_seq_length:
                warnings.warn((
                    'Found sentence with over {} tokens '
                    'sentence will be truncated to a valid length'
                ).format(self.max_seq_length))
                toks = toks[:self.max_seq_length]
            elif length < batch_length:
                toks += [self.pad_index] * (batch_length - length)
            return toks

        return torch.LongTensor([pad_toks(t) for t in tokens])

    def make_batches(
        self, texts: Iterator[str]
    ) -> Generator[BATCH_TYPE, None, None]:
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
                yield (
                    batch_indices,
                    batch_texts,
                    batch_tokens,
                    torch.LongTensor([len(t) for t in batch_tokens])
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
            yield (
                batch_indices,
                batch_texts,
                batch_tokens,
                torch.LongTensor([len(t) for t in batch_tokens])
            )
