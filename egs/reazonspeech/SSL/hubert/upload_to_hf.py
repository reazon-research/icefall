import argparse
import io
from dataclasses import dataclass, asdict, fields, MISSING
from pathlib import Path

import torch
from huggingface_hub import HfApi
from icefall.utils import AttributeDict

from finetune_ce import get_model


@dataclass(init=False)
class ModelParams:
    # Audio parameters
    label_rate: float = 50
    sample_rate: float = 16_000

    # Feature extractor parameters
    extractor_mode: str = "default"
    conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
    conv_bias: bool = False

    # Encoder parameters
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    layer_type: str = "transformer"

    # Dropout parameters
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0

    # Final projection parameters
    final_dim: int = 0
    untie_final_proj: bool = False
    layer_norm_first: bool = False

    # Logit and feature parameters
    logit_temp: float = 0.1
    target_glu: bool = False
    feature_grad_mult: float = 1.0

    # Masking parameters
    mask_length: int = 10
    mask_prob: float = 0.65
    mask_selection: str = "static"
    mask_other: float = 0
    no_mask_overlap: bool = False
    mask_min_space: int = 1

    # Channel masking parameters
    mask_channel_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_selection: str = "static"
    mask_channel_other: float = 0
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1

    # Positional embeddings
    conv_pos: int = 128
    conv_pos_groups: int = 16
    conv_pos_batch_norm: bool = False
    latent_temp: list = None

    # Loss computation parameters
    skip_masked: bool = False
    skip_nomask: bool = False
    checkpoint_activations: bool = False
    pred_masked_weight: float = 1
    pred_nomask_weight: float = 0
    loss_weights: list = None

    # FP16 optimization
    required_seq_len_multiple: int = 2

    # Attention type (for ESPnet compatibility)
    attn_type: str = ""
    pos_enc_type: str = "abs"

    # Number of classes
    num_classes: list = None

    def __post_init__(self):
        if self.latent_temp is None:
            self.latent_temp = [2, 0.5, 0.999995]
        if self.loss_weights is None:
            self.loss_weights = [10]
        if self.num_classes is None:
            self.num_classes = [504]

    def to_attribute_dict(self) -> AttributeDict:
        """Convert dataclass to AttributeDict for model initialization."""
        return AttributeDict(asdict(self))

    def __init__(self, checkpoint_data: dict):
        """Initialize params from checkpoint data.

        - Start with dataclass defaults
        - Override with keys from checkpoint_data["params"] if present
        - Print skipped keys that don't match attributes
        - Infer a few values from the model state dict when available
        """
        # Initialize all fields with their defaults
        for f in fields(ModelParams):
            if f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                setattr(self, f.name, f.default_factory())  # type: ignore[misc]
            else:
                setattr(self, f.name, None)

        # Override from checkpoint params if available
        saved_params = checkpoint_data.get("params", {})
        if isinstance(saved_params, dict):
            for key, value in saved_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"skip: {key}")

        # Infer from state dict if present
        state_dict = checkpoint_data.get("model", {})
        if isinstance(state_dict, dict):
            if "mask_emb" in state_dict and hasattr(state_dict["mask_emb"], "shape"):
                self.encoder_embed_dim = state_dict["mask_emb"].shape[0]

            if "final_proj.weight" in state_dict and hasattr(
                state_dict["final_proj.weight"], "shape"
            ):
                total_classes = state_dict["final_proj.weight"].shape[0]
                self.num_classes = [total_classes]

            encoder_layer_keys = [
                k for k in state_dict.keys() if "encoder.layers." in k
            ]
            if encoder_layer_keys:
                layer_nums = set()
                for key in encoder_layer_keys:
                    parts = key.split(".")
                    if len(parts) > 2 and parts[1] == "layers":
                        try:
                            layer_nums.add(int(parts[2]))
                        except ValueError:
                            pass
                if layer_nums:
                    self.encoder_layers = max(layer_nums) + 1

        # Finalize defaults for optional list fields
        self.__post_init__()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=Path, required=True)
    parser.add_argument("--upload-to", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    api = HfApi()
    data = torch.load(args.filepath, weights_only=False)
    model_state_dict = data["model"]

    # Initialize model parameters from checkpoint
    params = ModelParams(data)
    params_dict = params.to_attribute_dict()
    model = get_model(params_dict)
    model.load_state_dict(model_state_dict)

    api.create_repo(
        repo_id=args.upload_to, private=True, repo_type="model", exist_ok=True
    )

    # Upload model weight
    with io.BytesIO() as buf:
        torch.save(model_state_dict, buf)
        api.upload_file(
            path_or_fileobj=buf.getvalue(),
            path_in_repo="pytorch_model.bin",
            repo_id=args.upload_to,
            repo_type="model",
        )

    # TODO: Upload model params as json
    # TODO: Upload tokenizer


if __name__ == "__main__":
    args = get_args()
    main(args)
