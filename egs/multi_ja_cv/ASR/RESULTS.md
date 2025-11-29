## Results

### Zipformer

#### Non-streaming (Byte-Level BPE vocab_size=2000)

This recipe combines ReazonSpeech (Japanese) and CommonVoice English.

**Note: No trained models yet. All results are TBD.**

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
./zipformer/decode.py \
    --epoch 10 \
    --avg 1 \
    --exp-dir ./zipformer/exp \
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

Results will be added here after training.

#### Streaming (Byte-Level BPE vocab_size=2000)

This recipe combines ReazonSpeech (Japanese) and CommonVoice English.

**Note: No trained models yet. All results are TBD.**

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
TODO
```

To export the model with sherpa onnx:

```shell
./zipformer/export-onnx-streaming.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 10 \
  --avg 1 \
  --exp-dir ./zipformer/exp-streaming \
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
  --exp-dir ./zipformer/exp-streaming \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --tokens data/lang/bbpe_2000/tokens.txt \
  --epoch 10 \
  --avg 1 \
  --jit 1
```

You may also use decode chunk sizes `16`, `32`, `64`, `128`.

Results will be added here after training.
