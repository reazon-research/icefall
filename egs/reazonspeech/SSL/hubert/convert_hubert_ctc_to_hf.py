import contextlib
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn as nn
from finetune_ce import get_model, get_params, get_parser
from tokenizer import Tokenizer
from icefall.checkpoint import load_checkpoint
from transformers import (
    HubertForCTC,
    HubertConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)
from tokenizers.implementations import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

torch.load = partial(torch.load, map_location=torch.device("cpu"))


def main():
    parser = get_parser()
    Tokenizer.add_arguments(parser)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--upload-to", type=str, required=True)
    parser.add_argument("--average", action="store_true", default=False)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[])
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    sp = Tokenizer.load(args.lang, args.lang_type)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    params.use_transducer = False
    params.use_ctc = True

    model = get_model(params)
    # Load checkpoint
    if not args.average:
        _ = load_checkpoint(args.filename, model=model)
        state_dict = get_converted_state_dict(model)
    else:
        assert (args.start is not None and args.end is not None) or len(args.checkpoints) > 0
        checkpoints = []
        for checkpoint in Path(args.filename).parent.glob("checkpoint-*.pt"):
            step = int(checkpoint.name.removeprefix("checkpoint-").removesuffix(".pt"))
            if args.checkpoints:
                if step in args.checkpoints:
                    checkpoints.append(checkpoint)
            elif args.start <= step <= args.end:
                checkpoints.append(checkpoint)
        state_dict = None
        for checkpoint in checkpoints:
            _ = load_checkpoint(checkpoint, model=model)
            current_state_dict = get_converted_state_dict(model)
            if state_dict is None:
                state_dict = current_state_dict
            else:
                for n, p in current_state_dict.items():
                    state_dict[n] = state_dict[n] + p
        for n in state_dict.keys():
            state_dict[n] /= len(checkpoints)

    hf_model = HubertForCTC(config=HubertConfig(vocab_size=params.vocab_size))
    hf_model.load_state_dict(state_dict)

    extractor = Wav2Vec2FeatureExtractor()  # TODO: tempfile
    with NamedTemporaryFile(mode="w+", suffix=".json") as vocab_file, open(params.lang / "tokens.txt") if args.lang_type == "char" else contextlib.nullcontext() as rf:
        if args.lang_type == "char":
            data = [line.strip().split("\t") for line in rf.readlines()]
            json_data = {k: int(v) for k, v in data}
        else:
            tokenizer_model = args.lang / "bpe.model"
            spm_tokenizer = SentencePieceUnigramTokenizer.from_spm(tokenizer_model)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=spm_tokenizer._tokenizer,
                unk_token="<unk>",
                bos_token="<sos/eos>",
                eos_token="<sos/eos>",
                pad_token="<blk>",
            )
            json_data = {k: int(v) for k, v in tokenizer.get_vocab().items()}
        json.dump(json_data, vocab_file, ensure_ascii=False)
        vocab_file.seek(0)
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file=vocab_file.name,
            bos_token="<sos/eos>",
            eos_token="<sos/eos>",
            pad_token="<blk>",
        )
    processor = Wav2Vec2Processor(extractor, tokenizer)
    hf_model.push_to_hub(args.upload_to, private=True)
    processor.push_to_hub(args.upload_to, private=True)


def get_converted_state_dict(model: nn.Module) -> OrderedDict:
    state_dict = OrderedDict()
    for n, p in model.named_parameters():
        prefix = ""
        if n.startswith("encoder."):
            prefix = "hubert."
            n = n.removeprefix("encoder.")

        if n == "mask_emb":
            n = "masked_spec_embed"
        elif n.startswith("feature_extractor.conv_layers"):
            n = n.replace("0.weight", "conv.weight")
            n = n.replace("2.weight", "layer_norm.weight")
            n = n.replace("2.bias", "layer_norm.bias")
        elif n.startswith("post_extract_proj"):
            n = n.replace("post_extract_proj", "feature_projection.projection")
        elif n.startswith("encoder.pos_conv"):
            n = n.replace("encoder.pos_conv.0", "encoder.pos_conv_embed.conv")
        elif n.startswith("encoder.layers"):
            n = n.replace("self_attn.", "attention.")
            n = n.replace("self_attn_layer_norm.", "layer_norm.")
            n = n.replace("fc1", "feed_forward.intermediate_dense")
            n = n.replace("fc2", "feed_forward.output_dense")
        elif n.startswith("layer_norm"):
            n = n.replace("layer_norm", "feature_projection.layer_norm")
        elif n.startswith("ctc_output.1"):
            n = n.replace("ctc_output.1", "lm_head")

        if n.startswith("final_proj"):
            continue

        n = prefix + n
        state_dict[n] = p
    return state_dict


if __name__ == "__main__":
    main()
