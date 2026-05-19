# primejob examples

These are complete example projects. Each folder has its own `pyproject.toml` and `.python-version`; run commands from inside the example directory.

## Hugging Face JSON sentiment

```bash
cd examples/hf-sentiment-json
uv lock
primejob dataset push data --disk pj-hf-sentiment --subdir data
primejob run train.py --gpu CPU --disk pj-hf-sentiment --data-mode stage --plain --yes
```

`train.py` reads `PRIMEJOB_DATASET_PATH/data/train.json` and trains a tiny DistilBERT classifier with `transformers.Trainer`.

## Image folder classification

```bash
cd examples/image-folder-torch
uv lock
primejob dataset push data --disk pj-image-folder --subdir data
primejob run train.py --gpu CPU --disk pj-image-folder --data-mode stage --plain --yes
```

`train.py` reads `PRIMEJOB_DATASET_PATH/data/image_folder/<class>/*.ppm` and trains a tiny Torch CNN.

## Unsloth SFT

```bash
cd examples/unsloth-sft
uv lock
primejob run train.py --gpu H200 --country FI --plain --yes
```

This uses `unsloth/tinyllama-bnb-4bit` and runs two SFT steps on a four-row in-memory dataset. It is intended as a GPU/CUDA smoke test.
