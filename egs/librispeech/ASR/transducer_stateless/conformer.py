#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import warnings
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from transformer import Transformer

from icefall.utils import is_jit_tracing, make_pad_mask, subsequent_chunk_mask


class Conformer(Transformer):
    """
    Args:
        num_features (int): Number of input features
        output_dim (int): Number of output dimension
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimension
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        normalize_before (bool): whether to use layer_norm before the first block.
        vgg_frontend (bool): whether to use vgg frontend.
        dynamic_chunk_training (bool): whether to use dynamic chunk training, if
            you want to train a streaming model, this is expected to be True.
            When setting True, it will use a masking strategy to make the attention
            see only limited left and right context.
        short_chunk_threshold (float): a threshold to determinize the chunk size
            to be used in masking training, if the randomly generated chunk size
            is greater than ``max_len * short_chunk_threshold`` (max_len is the
            max sequence length of current batch) then it will use
            full context in training (i.e. with chunk size equals to max_len).
            This will be used only when dynamic_chunk_training is True.
        short_chunk_size (int): see docs above, if the randomly generated chunk
            size equals to or less than ``max_len * short_chunk_threshold``, the
            chunk size will be sampled uniformly from 1 to short_chunk_size.
            This also will be used only when dynamic_chunk_training is True.
        num_left_chunks (int): the left context (in chunks) attention can see, the
            chunk size is decided by short_chunk_threshold and short_chunk_size.
            A minus value means seeing full left context.
            This also will be used only when dynamic_chunk_training is True.
        causal (bool): Whether to use causal convolution in conformer encoder
            layer. This MUST be True when using dynamic_chunk_training.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        dynamic_chunk_training: bool = False,
        short_chunk_threshold: float = 0.75,
        short_chunk_size: int = 25,
        num_left_chunks: int = -1,
        causal: bool = False,
    ) -> None:
        super(Conformer, self).__init__(
            num_features=num_features,
            output_dim=output_dim,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            normalize_before=normalize_before,
            vgg_frontend=vgg_frontend,
        )

        self.encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal

        self.dynamic_chunk_training = dynamic_chunk_training
        self.short_chunk_threshold = short_chunk_threshold
        self.short_chunk_size = short_chunk_size
        self.num_left_chunks = num_left_chunks

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            cnn_module_kernel,
            normalize_before,
            causal,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_encoder_layers)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)
        else:
            # Note: TorchScript detects that self.after_norm could be used inside forward()
            #       and throws an error without this change.
            self.after_norm = identity

        self._init_state: List[torch.Tensor] = [torch.empty(0)]

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!

        #  lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        if not is_jit_tracing():
            assert x.size(0) == lengths.max().item()

        src_key_padding_mask = make_pad_mask(lengths)

        if self.dynamic_chunk_training:
            assert (
                self.causal
            ), "Causal convolution is required for streaming conformer."
            max_len = x.size(0)
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > (max_len * self.short_chunk_threshold):
                chunk_size = max_len
            else:
                chunk_size = chunk_size % self.short_chunk_size + 1

            mask = ~subsequent_chunk_mask(
                size=x.size(0),
                chunk_size=chunk_size,
                num_left_chunks=self.num_left_chunks,
                device=x.device,
            )
            x = self.encoder(
                x, pos_emb, mask=mask, src_key_padding_mask=src_key_padding_mask
            )  # (T, N, C)
        else:
            x = self.encoder(
                x, pos_emb, mask=None, src_key_padding_mask=src_key_padding_mask
            )  # (T, N, C)

        if self.normalize_before:
            x = self.after_norm(x)

        logits = self.encoder_output_layer(x)
        logits = logits.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return logits, lengths

    @torch.jit.export
    def get_init_state(
        self, left_context: int, device: torch.device
    ) -> List[torch.Tensor]:
        """Return the initial cache state of the model.

        Args:
          left_context: The left context size (in frames after subsampling).

        Returns:
          Return the initial state of the model, it is a list containing two
          tensors, the first one is the cache for attentions which has a shape
          of (num_encoder_layers, left_context, encoder_dim), the second one
          is the cache of conv_modules which has a shape of
          (num_encoder_layers, cnn_module_kernel - 1, encoder_dim).

          NOTE: the returned tensors are on the given device.
        """
        if len(self._init_state) == 2 and self._init_state[0].size(1) == left_context:
            # Note: It is OK to share the init state as it is
            # not going to be modified by the model
            return self._init_state

        init_states: List[torch.Tensor] = [
            torch.zeros(
                (
                    self.encoder_layers,
                    left_context,
                    self.d_model,
                ),
                device=device,
            ),
            torch.zeros(
                (
                    self.encoder_layers,
                    self.cnn_module_kernel - 1,
                    self.d_model,
                ),
                device=device,
            ),
        ]

        self._init_state = init_states

        return init_states

    @torch.jit.export
    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        processed_lens: Optional[Tensor] = None,
        left_context: int = 64,
        right_context: int = 0,
        chunk_size: int = 16,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            The decode states for previous frames which contains the cached data.
            It has two elements, the first element is the attn_cache which has
            a shape of (encoder_layers, left_context, batch, attention_dim),
            the second element is the conv_cache which has a shape of
            (encoder_layers, cnn_module_kernel-1, batch, conv_dim).
            Note: states will be modified in this function.
          processed_lens:
            How many frames (after subsampling) have been processed for each sequence.
          left_context:
            How many previous frames the attention can see in current chunk.
            Note: It's not that each individual frame has `left_context` frames
            of left context, some have more.
          right_context:
            How many future frames the attention can see in current chunk.
            Note: It's not that each individual frame has `right_context` frames
            of right context, some have more.
          chunk_size:
            The chunk size for decoding, this will be used to simulate streaming
            decoding using masking.
          simulate_streaming:
            If setting True, it will use a masking strategy to simulate streaming
            fashion (i.e. every chunk data only see limited left context and
            right context). The whole sequence is supposed to be send at a time
            When using simulate_streaming.
        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
            - states, the updated states(i.e. caches) including the information
              of current chunk.
        """

        # x: [N, T, C]
        # Caution: We assume the subsampling factor is 4!

        #  lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        if not simulate_streaming:
            assert states is not None
            assert processed_lens is not None
            assert (
                len(states) == 2
                and states[0].shape
                == (self.encoder_layers, left_context, x.size(0), self.d_model)
                and states[1].shape
                == (
                    self.encoder_layers,
                    self.cnn_module_kernel - 1,
                    x.size(0),
                    self.d_model,
                )
            ), f"""The length of states MUST be equal to 2, and the shape of
             first element should be {(self.encoder_layers, left_context, x.size(0), self.d_model)},
             given {states[0].shape}. the shape of second element should be
             {(self.encoder_layers, self.cnn_module_kernel - 1, x.size(0), self.d_model)},
             given {states[1].shape}."""

            lengths -= 2  # we will cut off 1 frame on each side of encoder_embed output
            src_key_padding_mask = make_pad_mask(lengths)

            processed_mask = torch.arange(left_context, device=x.device).expand(
                x.size(0), left_context
            )
            processed_lens = processed_lens.view(x.size(0), 1)
            processed_mask = (processed_lens <= processed_mask).flip(1)

            src_key_padding_mask = torch.cat(
                [processed_mask, src_key_padding_mask], dim=1
            )

            embed = self.encoder_embed(x)

            # cut off 1 frame on each size of embed as they see the padding
            # value which causes a training and decoding mismatch.
            embed = embed[:, 1:-1, :]

            embed, pos_enc = self.encoder_pos(embed, left_context)
            embed = embed.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)

            x, states = self.encoder.chunk_forward(
                embed,
                pos_enc,
                src_key_padding_mask=src_key_padding_mask,
                states=states,
                left_context=left_context,
                right_context=right_context,
            )  # (T, B, F)
        else:
            assert states is None
            states = []  # just to make torch.script.jit happy
            src_key_padding_mask = make_pad_mask(lengths)
            x = self.encoder_embed(x)
            x, pos_emb = self.encoder_pos(x)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

            assert x.size(0) == lengths.max().item()

            if chunk_size < 0:
                # use full attention
                chunk_size = x.size(0)
                left_context = -1

            num_left_chunks = -1
            if left_context >= 0:
                assert left_context % chunk_size == 0
                num_left_chunks = left_context // chunk_size

            mask = ~subsequent_chunk_mask(
                size=x.size(0),
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks,
                device=x.device,
            )
            x = self.encoder(
                x,
                pos_emb,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )  # (T, N, C)

        if self.normalize_before:
            x = self.after_norm(x)

        logits = self.encoder_output_layer(x)
        logits = logits.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return logits, lengths, states


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before (bool): whether to use layer_norm before the first block.
        causal (bool): Whether to use causal convolution in conformer encoder
            layer. This MUST be True when using dynamic_chunk_training and streaming decoding.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        causal: bool = False,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel, causal=causal)

        self.norm_ff_macaron = nn.LayerNorm(d_model)  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)

        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)

        # convolution module
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)

        src, _ = self.conv_module(src, src_key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src)

        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src

    @torch.jit.export
    def chunk_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        states: List[Tensor],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            states:
              The decode states for previous frames which contains the cached data.
              It has two elements, the first element is the attn_cache which has
              a shape of (left_context, batch, attention_dim),
              the second element is the conv_cache which has a shape of
              (cnn_module_kernel-1, batch, conv_dim).
              Note: states will be modified in this function.
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            left_context:
              How many previous frames the attention can see in current chunk.
              Note: It's not that each individual frame has `left_context` frames
              of left context, some have more.
            right_context:
              How many future frames the attention can see in current chunk.
              Note: It's not that each individual frame has `right_context` frames
              of right context, some have more.
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*(S+left_context)-1, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)

        # We put the attention cache this level (i.e. before linear transformation)
        # to save memory consumption, when decoding in streaming fashion, the
        # batch size would be thousands (for 32GB machine), if we cache key & val
        # separately, it needs extra several GB memory.
        # TODO(WeiKang): Move cache to self_attn level (i.e. cache key & val
        # separately) if needed.
        key = torch.cat([states[0], src], dim=0)
        val = key
        if right_context > 0:
            states[0] = key[
                -(left_context + right_context) : -right_context, ...  # noqa
            ]
        else:
            states[0] = key[-left_context:, ...]

        src_att = self.self_attn(
            src,
            key,
            val,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            left_context=left_context,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)

        # convolution module
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)

        src, conv_cache = self.conv_module(src, states[1], right_context=right_context)
        states[1] = conv_cache
        src = residual + self.dropout(src)

        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src, states


class ConformerEncoder(nn.Module):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for layer_index, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output

    @torch.jit.export
    def chunk_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        states: List[Tensor],
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            states:
              The decode states for previous frames which contains the cached data.
              It has two elements, the first element is the attn_cache which has
              a shape of (encoder_layers, left_context, batch, attention_dim),
              the second element is the conv_cache which has a shape of
              (encoder_layers, cnn_module_kernel-1, batch, conv_dim).
              Note: states will be modified in this function.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            left_context:
              How many previous frames the attention can see in current chunk.
              Note: It's not that each individual frame has `left_context` frames
              of left context, some have more.
            right_context:
              How many future frames the attention can see in current chunk.
              Note: It's not that each individual frame has `right_context` frames
              of right context, some have more.
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*(S+left_context)-1, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        assert not self.training
        output = src

        for layer_index, mod in enumerate(self.layers):
            cache = [states[0][layer_index], states[1][layer_index]]
            output, cache = mod.chunk_forward(
                output,
                pos_emb,
                states=cache,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                left_context=left_context,
                right_context=right_context,
            )
            states[0][layer_index] = cache[0]
            states[1][layer_index] = cache[1]

        return output, states


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        if is_jit_tracing():
            # 10k frames correspond to ~100k ms, e.g., 100 seconds, i.e.,
            # It assumes that the maximum input won't have more than
            # 10k frames.
            #
            # TODO(fangjun): Use torch.jit.script() for this module
            max_len = 10000

        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor, left_context: int = 0) -> None:
        """Reset the positional encodings."""
        x_size_1 = x.size(1) + left_context
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size_1 * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size_1, self.d_model)
        pe_negative = torch.zeros(x_size_1, self.d_model)
        position = torch.arange(0, x_size_1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, left_context: int = 0) -> Tuple[Tensor, Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x, left_context)
        x = x * self.xscale
        x_size_1 = x.size(1) + left_context
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x_size_1
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        left_context: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.num_heads,
            self.in_proj.weight,
            self.in_proj.bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            left_context=left_context,
        )

    def rel_shift(self, x: Tensor, left_context: int = 0) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape

        time2 = time1 + left_context
        if not is_jit_tracing():
            assert (
                n == left_context + 2 * time1 - 1
            ), f"{n} == {left_context} + 2 * {time1} - 1"

        if is_jit_tracing():
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(time2)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols

            x = x.reshape(-1, n)
            x = torch.gather(x, dim=1, index=indexes)
            x = x.reshape(batch_size, num_heads, time1, time2)
            return x
        else:
            # Note: TorchScript requires explicit arg for stride()
            batch_stride = x.stride(0)
            head_stride = x.stride(1)
            time1_stride = x.stride(2)
            n_stride = x.stride(3)
            return x.as_strided(
                (batch_size, num_heads, time1, time2),
                (batch_stride, head_stride, time1_stride - n_stride, n_stride),
                storage_offset=n_stride * (time1 - 1),
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        left_context: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        if not is_jit_tracing():
            assert embed_dim == embed_dim_to_check
            assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        if not is_jit_tracing():
            assert (
                head_dim * num_heads == embed_dim
            ), "embed_dim must be divisible by num_heads"

        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(
                3, dim=-1
            )

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        if not is_jit_tracing():
            assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)

        # (batch, 2*time1, head, d_k) --> (batch, head, d_k, 2*time -1)
        p = p.permute(0, 2, 3, 1)

        q_with_bias_u = (q + self.pos_bias_u).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self.pos_bias_v).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k)  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p)  # (batch, head, time1, 2*time1-1)

        matrix_bd = self.rel_shift(matrix_bd, left_context=left_context)

        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)

        if not is_jit_tracing():
            assert list(attn_output_weights.size()) == [
                bsz * num_heads,
                tgt_len,
                src_len,
            ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)

        # If we are using dynamic_chunk_training and setting a limited
        # num_left_chunks, the attention may only see the padding values which
        # will also be masked out by `key_padding_mask`, at this circumstances,
        # the whole column of `attn_output_weights` will be `-inf`
        # (i.e. be `nan` after softmax), so, we fill `0.0` at the masking
        # positions to avoid invalid loss value below.
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and key_padding_mask is not None
        ):
            combined_mask = attn_mask.unsqueeze(0) | key_padding_mask.unsqueeze(
                1
            ).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(combined_mask, 0.0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)

        if not is_jit_tracing():
            assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
        causal (bool): Whether to use causal convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.causal = causal

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.lorder = kernel_size - 1
        padding = (kernel_size - 1) // 2
        if self.causal:
            padding = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: Tensor,
        cache: Optional[Tensor] = None,
        right_context: int = 0,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: The cache of depthwise_conv, only used in real streaming
                decoding.
            right_context:
              How many future frames the attention can see in current chunk.
              Note: It's not that each individual frame has `right_context` frames
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        if self.causal and self.lorder > 0:
            if cache is None:
                # Make depthwise_conv causal by
                # manualy padding self.lorder zeros to the left
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                assert not self.training, "Cache should be None in training time"
                assert cache.size(0) == self.lorder
                x = torch.cat([cache.permute(1, 2, 0), x], dim=2)
                if right_context > 0:
                    cache = x.permute(2, 0, 1)[
                        -(self.lorder + right_context) : (-right_context),  # noqa
                        ...,
                    ]
                else:
                    cache = x.permute(2, 0, 1)[-self.lorder :, ...]  # noqa

        x = self.depthwise_conv(x)
        # x is (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        if cache is None:
            cache = torch.empty(0)

        return x.permute(2, 0, 1), cache


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def identity(x):
    return x
