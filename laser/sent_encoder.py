from typing import Optional, List

import numpy as np

import torch
from torch import nn

from laser.encoder import load_encoder_from_file
from laser.data import Batcher


class SentenceEncoder(nn.Module):

    def __init__(
        self,
        lang: str,
        model_path: str,
        bpe_codes: str,
        bpe_vocab: str,
        max_seq_length: int = 256,
        max_sents: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """Wrapper around Encoder and Batcher to encode sentences

        Arguments:
            lang {str} -- Language expected
                (this would only affect preprocessing)
            model_path {str} -- Path to model file
            bpe_codes {str} -- Path to bpe codes file
            bpe_vocab {str} -- Path to bpe vocab file

        Keyword Arguments:
            max_seq_length {int} --
                Maximum number of tokens per sentence (default: {256})
            max_sents {Optional[int]} --
                Maximum number of sentences per batch (default: {None})
            max_tokens {Optional[int]} --
                Maximum number of tokens per batch (default: {None})
        """
        super().__init__()
        self.encoder, dictionary = load_encoder_from_file(model_path)
        self.batcher = Batcher(
            lang=lang,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            dictionary=dictionary,
            max_seq_length=max_seq_length,
            max_sents=max_sents,
            max_tokens=max_tokens,
        )

        # Default load in eval mode
        self.eval()

    @torch.no_grad()
    def encode_sentences(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of sentences

        Arguments:
            texts {List[str]} -- A list of raw texts

        Returns:
            np.ndarray -- Sentence embeddings
        """
        device = self.encoder.embed_tokens.weight.device

        embeddings = [None] * len(texts)
        for batch in Batcher.make_batches(texts):
            outputs = self.encoder(
                batch.tokens.to(device),
                batch.lengths.to(device),
            ).sentemb.detach().cpu().numpy()

            for i, embedding in zip(batch.indices, outputs):
                embeddings[i] = embedding

        return np.stack(embeddings, axis=0)

    def forward(self, *args, **kwargs):
        """Alias for encode_sentences"""
        return self.encode_sentences(*args, **kwargs)
