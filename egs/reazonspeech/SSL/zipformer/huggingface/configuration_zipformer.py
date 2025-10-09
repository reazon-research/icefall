import os

import torch
from transformers import PretrainedConfig


def to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


class ZipformerConfig(PretrainedConfig):
    model_type = "zipformer"

    @classmethod
    def from_icefall_checkpoint(cls, checkpoint_path: os.PathLike, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "vocab_size" in kwargs:
            kwargs.update({"vocab_size": kwargs["vocab_size"]})
        return cls(
            conv_feature_layers=eval(checkpoint["conv_feature_layers"]),
            extractor_mode=checkpoint["extractor_mode"],
            conv_bias=checkpoint["conv_bias"],
            encoder_dim=to_int_tuple(checkpoint["encoder_dim"]),
            dropout_input=checkpoint["dropout_input"],
            dropout_features=checkpoint["dropout_features"],
            feature_grad_mult=checkpoint["feature_grad_mult"],
            downsampling_factor=to_int_tuple(checkpoint["downsampling_factor"]),
            num_encoder_layers=to_int_tuple(checkpoint["num_encoder_layers"]),
            encoder_unmasked_dim=to_int_tuple(checkpoint["encoder_unmasked_dim"]),
            query_head_dim=checkpoint["query_head_dim"],
            pos_head_dim=checkpoint["pos_head_dim"],
            value_head_dim=checkpoint["value_head_dim"],
            pos_dim=checkpoint["pos_dim"],
            num_heads=to_int_tuple(checkpoint["num_heads"]),
            feedforward_dim=to_int_tuple(checkpoint["feedforward_dim"]),
            cnn_module_kernel=to_int_tuple(checkpoint["cnn_module_kernel"]),
            **kwargs,
        )

    def __init__(
        self,
        conv_feature_layers: list[tuple[int, int, int]] = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_mode: str = "default",
        conv_bias: bool = False,
        encoder_dim: list[int] = [192, 256, 448, 768, 448, 192],
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        feature_grad_mult: float = 1.0,
        downsampling_factor: list[int] = [1, 2, 4, 8, 4, 2],
        num_encoder_layers: list[int] = [2, 2, 3, 4, 3, 2],
        encoder_unmasked_dim: list[int] = [192, 192, 256, 256, 256, 192],
        query_head_dim: int = 32,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        pos_dim: int = 48,
        num_heads: list[int] = [4, 4, 4, 8, 4, 4],
        feedforward_dim: list[int] = [512, 768, 1024, 1536, 1024, 768],
        cnn_module_kernel: list[int] = [31, 31, 15, 15, 15, 31],
        vocab_size: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_unmasked_dim = encoder_unmasked_dim
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.value_head_dim = value_head_dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.cnn_module_kernel = cnn_module_kernel
        self.downsampling_factor = downsampling_factor
        self.num_encoder_layers = num_encoder_layers
        self.encoder_unmasked_dim = encoder_unmasked_dim
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.value_head_dim = value_head_dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.cnn_module_kernel = cnn_module_kernel
        self.vocab_size = vocab_size
