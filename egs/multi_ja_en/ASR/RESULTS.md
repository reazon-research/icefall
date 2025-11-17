## Results

### Zipformer

#### Non-streaming (Byte-Level BPE vocab_size=2000)

Trained on 15k hours of ReazonSpeech (filtered to only audio segments between 8s and 22s) and 15k hours of MLS English.

The training command is:

```shell
./zipformer/train.py \
  --world-size 8 \
  --causal 0 \
  --num-epochs 10 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --manifest-dir data/manifests \
  --enable-musan True
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 10 \
    --avg 1 \
    --exp-dir ./zipformer/exp \
    --bpe-model data/lang/bbpe_2000/bbpe.model \
    --decoding-method modified_beam_search \
    --manifest-dir data/manifests
```

To export the model with onnx:

```shell
./zipformer/export-onnx.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 10 \
  --avg 1 \
  --exp-dir ./zipformer/exp
```

WER on test sets listed below (calculated with `./zipformer/decode.py`):

| Decoding Method      | Combined | ReazonSpeech (Japanese) | MLS English |
|----------------------|----------|-------------------------|-------------|
| greedy_search        | 7.10     | 6.05                    | 8.76        |
| modified_beam_search | 6.32     | TBD                     | TBD         |

Note: Japanese text is tokenized character-level via `tokenize_by_ja_char`, so WER is effectively character error rate (CER).



We also include WER% for common English ASR datasets:

| Corpus                      | WER (%) |
|-----------------------------|---------|
| CommonVoice                 | 29.03   |
| TED                         | 16.78   |
| MLS English (test set)      | 8.64    |


And CER% for common Japanese datasets:

| Corpus        | CER (%) |
|---------------|---------|
| JSUT          | 8.13   |
| CommonVoice   | 9.82   |
| TEDx          | 11.64   |


Pre-trained model can be found here: [https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k](https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k)

(Not yet publicly released)

#### Streaming (Byte-Level BPE vocab_size=2000)

Trained on 15k hours of ReazonSpeech (filtered to only audio segments between 8s and 22s) and 15k hours of MLS English.

The training command is:

```shell
./zipformer/train.py \
  --world-size 8 \
  --causal 1 \
  --num-epochs 10 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --manifest-dir data/manifests \
  --enable-musan True
```

The decoding command is:

```shell
./zipformer/streaming_decode.py \
  --epoch 23 \
  --avg 1 \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --exp-dir ./zipformer/exp-15k15k-streaming \
  --bpe-model data/lang/bbpe_2000/bbpe.model \
  --decoding-method greedy_search \
  --num-decode-streams 2000 \
  --manifest-dir data/manifests
```

To export the model with sherpa onnx:

```shell
./zipformer/export-onnx-streaming.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 10 \
  --avg 1 \
  --exp-dir ./zipformer/exp-15k15k-streaming \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 128 \
  --fp16 True
```

(Adjust the `chunk-size` and `left-context-frames` as necessary)

To export the model as Torchscript (`.jit`):

```shell
./zipformer/export.py \
  --exp-dir ./zipformer/exp-15k15k-streaming \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --tokens data/lang/bbpe_2000/tokens.txt \
  --epoch 10 \
  --avg 1 \
  --jit 1
```

You may also use decode chunk sizes `16`, `32`, `64`, `128`.

WER on test sets listed below (calculated with `./zipformer/streaming_decode.py` with chunk-size 16, left-context-frames 128):

| Decoding Method | Combined | ReazonSpeech (Japanese) | MLS English |
|-----------------|----------|-------------------------|-------------|
| greedy_search   | 9.77     | 8.66                    | 11.52       |

Note: Japanese text is tokenized character-level via `tokenize_by_ja_char`, so WER is effectively character error rate (CER).


We also include WER% for common English ASR datasets:

*TODO: Evaluate with streaming_decode.py*


And CER% for common Japanese datasets:

*TODO: Evaluate with streaming_decode.py*


Pre-trained model can be found here: [https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k](https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k)

(Not yet publicly released)
