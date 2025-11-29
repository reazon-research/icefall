#!/bin/bash
cd /root/Github/reazon-icefall/egs/multi_ja_en/ASR
python zipformer/streaming_decode.py \
  --epoch 23 \
  --avg 1 \
  --use-averaged-model 0 \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --exp-dir ./zipformer/exp-streaming-eval \
  --bpe-model data/lang/bbpe_2000/bbpe.model \
  --decoding-method greedy_search \
  --num-decode-streams 2000 \
  --manifest-dir data/manifests
