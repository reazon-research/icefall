from collections import OrderedDict

import torch.nn as nn
from pretrain_ce import get_model, get_params, get_parser
from icefall.checkpoint import load_checkpoint
from transformers import HubertModel, HubertConfig


def main():
    parser = get_parser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--upload-to", type=str, required=True)

    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    model = get_model(params)
    # Load checkponint
    _ = load_checkpoint(args.filename, model=model)
    state_dict = get_converted_state_dict(model)

    hf_model = HubertModel(config=HubertConfig())
    hf_model.load_state_dict(state_dict)

    hf_model.push_to_hub(args.upload_to, private=True)


def get_converted_state_dict(model: nn.Module) -> OrderedDict:
    state_dict = OrderedDict()
    for n, p in model.named_parameters():
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

        if n.startswith("final_proj"):
            continue

        state_dict[n] = p
    return state_dict


if __name__ == "__main__":
    main()
