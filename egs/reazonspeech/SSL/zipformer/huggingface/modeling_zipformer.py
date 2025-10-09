from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, Wav2Vec2BaseModelOutput

from configuration_zipformer import ZipformerConfig
from wav2vec2_module import ConvFeatureExtractionModel
from scaling import ScheduledFloat
from utils import GradMultiply, LayerNorm
from zipformer import Zipformer2


class ZipformerModel(PreTrainedModel):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__(config)
        feature_enc_layers = config.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=config.extractor_mode,
            conv_bias=config.conv_bias,
        )
        encoder_input_dim = config.encoder_dim[0]
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_input_dim)
            if self.embed != encoder_input_dim
            else None
        )

        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)

        self.feature_grad_mult = config.feature_grad_mult

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_input_dim).uniform_())

        self.encoder = Zipformer2(
            output_downsampling_factor=1,
            downsampling_factor=config.downsampling_factor,
            num_encoder_layers=config.num_encoder_layers,
            encoder_dim=config.encoder_dim,
            encoder_unmasked_dim=config.encoder_unmasked_dim,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            pos_dim=config.pos_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            warmup_batches=4000.0,
        )

        self.layer_norm = LayerNorm(self.embed)

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Wav2Vec2BaseModelOutput:
        """output layer is 1-based"""
        features = self.forward_features(input_values)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float -> (T, B, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x = x.transpose(0, 1)
        x, _ = self.encoder(x, (~padding_mask).sum(dim=-1))
        x = x.transpose(0, 1)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=x,
            extract_features=features,
        )


class ZipformerForCTC(PreTrainedModel):
    def __init__(self, config: ZipformerConfig):
        super().__init__(config)
        self.encoder = ZipformerModel(config)
        self.ctc_output = nn.Linear(config.encoder_dim[-1], config.vocab_size)

    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        if padding_mask is None:
            padding_mask = torch.zeros_like(input_values, dtype=torch.bool)

        encoder_out = self.encoder(input_values, padding_mask)
        ctc_output = self.ctc_output(encoder_out.last_hidden_state)

        return CausalLMOutput(
            loss=None, logits=ctc_output, hidden_states=None, attentions=None
        )
