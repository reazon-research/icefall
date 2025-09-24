import argparse
import io
import json
import re
from dataclasses import dataclass, asdict, fields, MISSING
from pathlib import Path
from typing import List

import torch
from huggingface_hub import HfApi
from icefall.utils import AttributeDict

from finetune_ce import get_model
from tokenizer import Tokenizer


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

    # ASR model params
    decoder_dim: int = 512
    joiner_dim: int = 512
    use_ctc: bool = False
    use_transducer: bool = True
    lm_scale: float = 0.25
    am_scale: float = 0.0
    simple_loss_scale: float = 0.5
    ctc_loss_scale: float = 0.2
    prune_range: int = 5
    context_size: int = 2
    blank_id: int = 0
    vocab_size: int = 504
    num_classes: list = None

    # Tokenizer params
    lang: Path = None
    lang_type: str = None

    # Feature extractor params
    do_normalize: bool = False

    def __post_init__(self):
        if self.latent_temp is None:
            self.latent_temp = [2, 0.5, 0.999995]
        if self.loss_weights is None:
            self.loss_weights = [10]
        if self.num_classes is None:
            self.num_classes = [self.vocab_size]

    def to_dict(self) -> dict:
        """Convert dataclass to dict."""
        return asdict(self)

    def to_attribute_dict(self) -> AttributeDict:
        """Convert dataclass to AttributeDict for model initialization."""
        return AttributeDict(self.to_dict())

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

        for key, value in checkpoint_data.items():
            if key == "params":
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Print skipped keys that are not attributes (e.g., model, optimizer)
                print(f"skip: {key}")

        # Infer from state dict if present
        state_dict = checkpoint_data.get("model", {})
        if isinstance(state_dict, dict):
            if "mask_emb" in state_dict and hasattr(state_dict["mask_emb"], "shape"):
                self.encoder_embed_dim = state_dict["mask_emb"].shape[0]

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


def find_matching_files(pattern: str) -> List[Path]:
    parent_dir = Path(pattern).parent
    filename_regex = re.compile(pattern=pattern.split("/")[-1])
    return sorted(
        [file for file in parent_dir.glob("*") if filename_regex.match(file.name)]
    )


def average_model_weights(file_paths: List[Path]) -> dict:
    """Average model state dicts from multiple checkpoint files."""
    if not file_paths:
        raise ValueError("No files provided for averaging")

    if len(file_paths) == 1:
        # Single file, no averaging needed
        data = torch.load(
            file_paths[0], weights_only=False, map_location=torch.device("cpu")
        )
        return data

    print(f"Averaging {len(file_paths)} checkpoint files:")
    for fp in file_paths:
        print(f"  - {fp}")

    # Load first checkpoint as reference
    first_data = torch.load(
        file_paths[0], weights_only=False, map_location=torch.device("cpu")
    )
    averaged_state_dict = {}

    # Initialize averaged state dict with zeros
    for key, tensor in first_data["model"].items():
        averaged_state_dict[key] = torch.zeros_like(tensor, dtype=torch.float32)

    # Sum all model weights
    for file_path in file_paths:
        data = torch.load(
            file_path, weights_only=False, map_location=torch.device("cpu")
        )
        model_state_dict = data["model"]

        for key, tensor in model_state_dict.items():
            if key in averaged_state_dict:
                averaged_state_dict[key] += tensor.float()
            else:
                print(f"Warning: Key {key} not found in first checkpoint, skipping")

    # Average the weights
    num_files = len(file_paths)
    for key in averaged_state_dict:
        averaged_state_dict[key] /= num_files
        # Convert back to original dtype
        averaged_state_dict[key] = averaged_state_dict[key].to(
            first_data["model"][key].dtype
        )

    # Use the first checkpoint's metadata with averaged model weights
    result_data = first_data.copy()
    result_data["model"] = averaged_state_dict

    return result_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path or regex pattern to match checkpoint files",
    )
    parser.add_argument("--upload-to", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    api = HfApi()

    # Check if filepath is a regex pattern or single file
    if Path(args.filepath).exists():
        # Single file path
        matching_files = [Path(args.filepath)]
    else:
        # Treat as regex pattern
        matching_files = find_matching_files(args.filepath)
        if not matching_files:
            raise ValueError(f"No files found matching pattern: {args.filepath}")

    # Load and potentially average checkpoints
    data = average_model_weights(matching_files)
    model_state_dict = data["model"]
    sp = Tokenizer.load(data["lang"], data["lang_type"])
    data["blank_id"] = sp.piece_to_id("<blk>")
    data["vocab_size"] = sp.get_piece_size()

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

    # Upload model params as json
    params_dict = {
        k: v for k, v in params.to_dict().items() if k not in ("lang", "lang_type")
    }
    api.upload_file(
        path_or_fileobj=json.dumps(params_dict, indent=4).encode("utf-8"),
        path_in_repo="config.json",
        repo_id=args.upload_to,
        repo_type="model",
    )

    # Upload tokenizer
    tokenizer_path = params.lang
    api.upload_folder(
        repo_id=args.upload_to,
        folder_path=tokenizer_path,
        path_in_repo="lang",
        repo_type="model",
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
