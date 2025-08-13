import argparse
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Optional

import fairseq
import joblib
import lhotse
import multiprocess
import numpy as np
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.utils import fastcopy
from tqdm import tqdm

torch.load = partial(torch.load, weights_only=False)  # HACK
lhotse.set_ffmpeg_torchaudio_info_enabled(True)  # Some flac files are broken, but ffmpeg can read them

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Global for multiprocess
apply_kmeans = None
model = None
do_normalize = True


class ApplyKmeans(object):
    def __init__(self, km_path, device=None):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if device is not None:
            self.C = self.C.to(device)
            self.Cnorm = self.Cnorm.to(device)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--part", type=str, required=True)
    parser.add_argument(
        "--model-path", type=str, default="download/hubert_base_ls960.pt"
    )
    parser.add_argument(
        "--kmeans-model-path",
        type=str,
        default="download/hubert_base_ls960_L9_km500.bin",
    )
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Number of processes to use."
    )

    parser.add_argument(
        "--window-duration",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--shift-duration",
        type=float,
        default=250.0,
    )

    return parser.parse_args()


@torch.no_grad()
def extract_and_save_one_cuts(
    raw_cuts_path: str,
    cuts_path: str,
    model_path: str,
    kmeans_model_path: str,
    window_duration: float,
    shift_duration: float,
    num_proc: int,
):
    logging.info(f"Loading {raw_cuts_path}")
    cut_set = CutSet.from_file(raw_cuts_path)

    logging.info("Extracting kmeans")
    cuts: list[CutSet] = []

    shards: list[CutSet] = cut_set.split(num_splits=num_proc)
    inputs = [
        [
            rank,
            shard,
            model_path,
            kmeans_model_path,
            window_duration,
            shift_duration,
        ]
        for rank, shard in enumerate(shards)
    ]
    logging.info("Created shards of CutSet")

    if num_proc > 1:  # For multi-gpu
        multiprocess.set_start_method("spawn", force=True)
        logging.info("Spawn processes")

    with multiprocess.Pool(num_proc) as pool:
        for res in pool.imap(
            lambda x: extract_and_save_one_cuts_single_process(*x),
            inputs,
        ):
            cuts.append(res)

    cuts = lhotse.combine(cuts)
    logging.info(f"Saving to {cuts_path}")
    cuts.to_file(cuts_path)


@torch.no_grad()
def extract_and_save_one_cuts_single_process(
    rank: int,
    cut_set: CutSet,
    model_path: str,
    kmeans_model_path: str,
    window_duration: float,
    shift_duration: float,
    num_gpus: int | None = None,
) -> CutSet:
    global model, apply_kmeans, do_normalize

    device = torch.device("cuda", rank % (num_gpus or torch.cuda.device_count()))
    if model is None:
        model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )
        model = model[0].eval().to(device)
        do_normalize = task.cfg.normalize
    if apply_kmeans is None:
        apply_kmeans = ApplyKmeans(kmeans_model_path, device=device)

    assert window_duration >= shift_duration
    window_size = int(window_duration * 16000)
    shift_size = int(shift_duration * 16000)
    overlap_size = window_size - shift_size
    out_overlap_size = get_out_length(overlap_size)

    cuts = []

    for cut in tqdm(
        cut_set,
        desc=f"Extracting ({rank=}, {device})",
        position=rank,
        dynamic_ncols=True
    ):
        assert cut.sampling_rate == 16000, f"Sampling rate: {cut.sampling_rate}"

        audio = cut.load_audio()

        T = audio.shape[1]
        start = 0
        kmeans = []
        while start < T:
            real_window_size = min(window_size, T - start)
            audio_window = audio[:, start : start + real_window_size]

            x = (
                torch.from_numpy(audio_window)
                .float()
                .to(next(model.parameters()).device)
            )
            if do_normalize:
                x = torch.nn.functional.layer_norm(x, x.shape)

            feature, _ = model.extract_features(
                source=x,
                padding_mask=None,
                mask=False,
                output_layer=9,
            )
            feature = feature.squeeze(0)

            current_kmeans = apply_kmeans(feature).tolist()

            if start == 0:
                kmeans.extend(current_kmeans)
            else:
                kmeans.extend(current_kmeans[out_overlap_size:])

            if T - start <= window_size:
                break

            start += shift_size

        kmeans = " ".join(map(str, kmeans))

        cut_with_kmeans = fastcopy(
            cut,
            custom={"kmeans": kmeans},
        )
        cuts.append(cut_with_kmeans)

    cuts = CutSet(cuts)

    return cuts


def extract_kmeans(args):
    output_dir = Path(args.output_dir) / args.part
    assert output_dir.exists(), f"{output_dir} does not exist!"

    prefix = "reazonspeech"
    dataset_splits = ["test", "dev", "train"]

    window_duration = args.window_duration
    shift_duration = args.shift_duration

    for dataset_split in dataset_splits:
        cuts_path = output_dir / f"{prefix}_cuts_{dataset_split}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = output_dir / f"{prefix}_cuts_{dataset_split}_raw.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        extract_and_save_one_cuts(
            raw_cuts_path=raw_cuts_path,
            cuts_path=cuts_path,
            model_path=args.model_path,
            kmeans_model_path=args.kmeans_model_path,
            window_duration=window_duration,
            shift_duration=shift_duration,
            num_proc=args.num_proc if dataset_split == "train" else 1,
        )


def get_out_length(T):
    conv_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    for i, (out_channels, kernel_size, stride) in enumerate(conv_layers):
        T = math.floor((T - kernel_size) / stride) + 1

    return max(0, T)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    extract_kmeans(args)
