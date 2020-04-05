"""LASER bilstm encoder
"""

import torch
from torch import nn
from typing import NamedTuple, Optional, Tuple


class EncoderOuts(NamedTuple):
    sentemb: torch.Tensor
    encoder_out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    encoder_padding_mask: Optional[torch.Tensor]


class Encoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        padding_value=0.
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim,
            padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = \
            self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([
                        outs[2 * i],
                        outs[2 * i + 1]
                    ], dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return EncoderOuts(
            sentemb=sentemb,
            encoder_out=(x, final_hiddens, final_cells),
            encoder_padding_mask=encoder_padding_mask
        )


class EncoderNew(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        padding_idx: int,
        embed_dim: int = 320,
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = False,
        padding_value: float = 0.0
    ):
        """LSTM based sequence encoder

        Arguments:
            num_embeddings {int} -- Number of unique token embeddings
            padding_idx {int} -- Padding token id

        Keyword Arguments:
            embed_dim {int} -- Embedding dimension size (default: {320})
            hidden_size {int} -- Hidden layer dimension size (default: {512})
            num_layers {int} -- Number of LSTM layers (default: {1})
            bidirectional {bool} -- Use Bidirectional LSTM (default: {False})
            padding_value {float} --
                Value to pad hidden layers (default: {0.0})
        """
        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.padding_idx = int(padding_idx)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.padding_value = float(padding_value)

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )

        self.num_directions = 2 if self.bidirectional else 1
        self.output_units = self.num_directions * self.hidden_size

    def forward(
        self,
        src_tokens: torch.LongTensor,
        src_lengths: torch.LongTensor,
        return_encoder_out: bool = False,
        return_encoder_padding_mask: bool = False,
    ) -> EncoderOuts:
        """Encode a batch of sequences

        Arguments:
            src_tokens {torch.LongTensor} -- [batch_size, seq_len]
            src_lengths {torch.LongTensor} -- [batch_size]

        Keyword Arguments:
            return_encoder_out {bool} --
                Return output tensors? (default: {False})
            return_encoder_padding_mask {bool} --
                Return encoder padding mask? (default: {False})

        Returns:
            [type] -- [description]
        """
        bsz, seqlen = src_tokens.size()

        x = self.embed_tokens(src_tokens)
        x = x.transpose(0, 1)  # BTC -> TBC

        # Pack then apply LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths, batch_first=False, enforce_sorted=True)
        packed_outs, (final_hiddens, final_cells) = \
            self.lstm.forward(packed_x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t()
        if padding_mask.any():
            x = x.float().masked_fill_(
                mask=padding_mask.unsqueeze(-1),
                value=float('-inf'),
            ).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        encoder_out = None
        if return_encoder_out:
            if self.bidirectional:
                final_hiddens = self._combine_outs(final_hiddens)
                final_cells = self._combine_outs(final_cells)
            encoder_out = (x, final_hiddens, final_cells)

        encoder_padding_mask = None
        if return_encoder_padding_mask:
            encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return EncoderOuts(
            sentemb=sentemb,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask
        )

    def _combine_outs(self, x: torch.Tensor) -> torch.Tensor:
        """Combines outputs for same layer same entry in batch.
        Only used for BiLSTM.

        Arguments:
            outs {torch.Tensor} -- [num_layers * num_dir, bsz, hidden_size]

        Returns:
            torch.Tensor -- [num_layers, bsz, num_dir * hidden_size]
        """
        # [num_layers * num_dir, bsz, hidden_size]
        #   -> [num_layers, num_dir, bsz, hidden_size]
        #   -> [num_layers, bsz, num_dir, hidden_size]
        #   -> [num_layers, bsz, num_dir * hidden_size]
        x = x.reshape(
            self.num_layers,
            self.num_directions,
            -1,
            self.hidden_size
        ).transpose(1, 2)
        x = x.transpose(1, 2)
        return x.reshape(self.num_layers, -1, self.output_units)


def load_model_from_file(model_path: str):
    # Load state_dict
    state_dict = torch.load(model_path, map_location='cpu')
    if 'left_pad' in state_dict['params']:
        del state_dict['params']['left_pad']

    # Create encoder
    encoder = Encoder(**state_dict['params'])
    encoder.load_state_dict(state_dict['model'])

    return encoder, state_dict['dictionary']
