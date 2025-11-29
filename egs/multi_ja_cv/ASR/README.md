# Introduction

A bilingual Japanese-English ASR model that utilizes ReazonSpeech (Japanese) and CommonVoice English datasets.

**ReazonSpeech** is an open-source dataset that contains a diverse set of natural Japanese speech, collected from terrestrial television streams. It contains more than 35,000 hours of audio.

**CommonVoice** is Mozilla's initiative to help teach machines how real people speak. It contains crowd-sourced voice data in multiple languages. The English dataset contains diverse speakers and accents from around the world.


# Training Sets

1. ReazonSpeech (Japanese)
2. CommonVoice (English)

|Dataset| Number of hours| URL|
|---|---:|---|
|CommonVoice English|~2,500|https://commonvoice.mozilla.org/|
|ReazonSpeech (all)|35,000|https://huggingface.co/datasets/reazon-research/reazonspeech|

# Usage

This recipe relies on the `commonvoice` recipe and the `reazonspeech` recipe.

To be able to use the `multi_ja_cv` recipe, you must first run the `prepare.sh` scripts in both the `commonvoice` recipe (with `--lang en`) and the `reazonspeech` recipe.

This recipe does not enforce data balance: please ensure that the `commonvoice` English and `reazonspeech` datasets prepared above are balanced to your liking.

Steps for model training:

0. Run `../../commonvoice/ASR/prepare.sh --lang en --stage 0 --stop-stage 5` and `../../reazonspeech/ASR/prepare.sh`
1. Run `./prepare.sh`
2. Run `python local/utils/update_cutset_paths.py`
3. Run `zipformer/train.py` (see example arguments inside the file)


