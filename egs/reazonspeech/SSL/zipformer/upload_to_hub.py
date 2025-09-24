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

from finetune import get_model
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

    # Dropout and logits
    dropout_input: float = 0.0
    dropout_features: float = 0.0
    logit_temp: float = 0.1
    skip_masked: bool = False
    skip_nomask: bool = False

    # Zipformer2 topology (defaults adapted from finetune.py)
    num_encoder_layers: str = "2,2,3,4,3,2"
    downsampling_factor: str = "1,2,4,8,4,2"
    feedforward_dim: str = "512,768,1024,1536,1024,768"
    num_heads: str = "4,4,4,8,4,4"
    encoder_dim: str = "192,256,384,512,384,256"
    query_head_dim: str = "32"
    value_head_dim: str = "12"
    pos_head_dim: str = "4"
    pos_dim: int = 48
    encoder_unmasked_dim: str = "192,192,256,256,256,192"
    cnn_module_kernel: str = "31,31,15,15,15,31"

    # Final projection and SSL loss
    untie_final_proj: bool = False
    pred_masked_weight: float = 1.0
    pred_nomask_weight: float = 0.0
    loss_weights: list = None

    # ASR model params
    decoder_dim: int = 512
    joiner_dim: int = 512
    use_ctc: bool = False
    use_transducer: bool = True
    lm_scale: float = 0.25
    am_scale: float = 0.0
    simple_loss_scale: float = 0.5
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
        if self.loss_weights is None:
            self.loss_weights = [10]
        if self.num_classes is None:
            self.num_classes = [self.vocab_size]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_attribute_dict(self) -> AttributeDict:
        return AttributeDict(self.to_dict())

    def __init__(self, checkpoint_data: dict):
        # Initialize defaults
        for f in fields(ModelParams):
            if f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                setattr(self, f.name, f.default_factory())  # type: ignore[misc]
            else:
                setattr(self, f.name, None)

        # Overlay from checkpoint top-level keys (excluding 'params')
        for key, value in checkpoint_data.items():
            if key == "params":
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Report unknown keys for clarity
                print(f"skip: {key}")

        self.__post_init__()


def find_matching_files(pattern: str) -> List[Path]:
    parent_dir = Path(pattern).parent
    filename_regex = re.compile(pattern=pattern.split("/")[-1])
    return [file for file in parent_dir.glob("*") if filename_regex.match(file.name)]


def average_model_weights(file_paths: List[Path]) -> dict:
    if not file_paths:
        raise ValueError("No files provided for averaging")

    if len(file_paths) == 1:
        data = torch.load(
            file_paths[0], weights_only=False, map_location=torch.device("cpu")
        )
        return data

    print(f"Averaging {len(file_paths)} checkpoint files:")
    for fp in file_paths:
        print(f"  - {fp}")

    first_data = torch.load(
        file_paths[0], weights_only=False, map_location=torch.device("cpu")
    )
    averaged_state = {}
    for k, t in first_data["model"].items():
        averaged_state[k] = torch.zeros_like(t, dtype=torch.float32)

    for fp in file_paths:
        d = torch.load(fp, weights_only=False, map_location=torch.device("cpu"))
        for k, t in d["model"].items():
            if k in averaged_state:
                averaged_state[k] += t.float()
            else:
                print(f"Warning: Key {k} not in reference, skipping")

    n = len(file_paths)
    for k in averaged_state:
        averaged_state[k] /= n
        averaged_state[k] = averaged_state[k].to(first_data["model"][k].dtype)

    out = first_data.copy()
    out["model"] = averaged_state
    return out


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath", type=str, required=True, help="Path or regex for checkpoint(s)"
    )
    parser.add_argument("--upload-to", type=str, required=True)
    return parser.parse_args()


def main(args):
    api = HfApi()

    if Path(args.filepath).exists():
        files = [Path(args.filepath)]
    else:
        files = find_matching_files(args.filepath)
        if not files:
            raise ValueError(f"No files match: {args.filepath}")

    data = average_model_weights(files)
    model_state = data["model"]
    sp = Tokenizer.load(data["lang"], data["lang_type"])
    data["blank_id"] = sp.piece_to_id("<blk>")
    data["vocab_size"] = sp.get_piece_size()

    params = ModelParams(data)
    model = get_model(params.to_attribute_dict())
    model.load_state_dict(model_state)

    api.create_repo(
        repo_id=args.upload_to, private=True, repo_type="model", exist_ok=True
    )

    # Upload model weights
    with io.BytesIO() as buf:
        torch.save(model_state, buf)
        api.upload_file(
            path_or_fileobj=buf.getvalue(),
            path_in_repo="pytorch_model.bin",
            repo_id=args.upload_to,
            repo_type="model",
        )

    # Upload parameters (excluding tokenizer paths)
    params_json = {
        k: v for k, v in params.to_dict().items() if k not in ("lang", "lang_type")
    }
    api.upload_file(
        path_or_fileobj=json.dumps(params_json, indent=4).encode("utf-8"),
        path_in_repo="config.json",
        repo_id=args.upload_to,
        repo_type="model",
    )

    # Upload tokenizer folder
    api.upload_folder(
        repo_id=args.upload_to,
        folder_path=params.lang,
        path_in_repo="lang",
        repo_type="model",
    )


if __name__ == "__main__":
    main(get_args())
