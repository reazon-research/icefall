import argparse
import io
from pathlib import Path

import torch
from huggingface_hub import HfApi


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=Path, required=True)
    parser.add_argument("--upload-to", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    api = HfApi()
    data = torch.load(args.filepath, weights_only=False)
    model = data["model"]
    # TODO: parse model params and convert into json

    # TODO: sanity check

    api.create_repo(repo_id=args.upload_to, private=True, repo_type="model", exist_ok=True)

    # Upload model weight
    with io.BytesIO() as buf:
        torch.save(model, buf)
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
