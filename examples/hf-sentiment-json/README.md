# Hugging Face JSON sentiment

Small end-to-end `primejob` example using a JSON dataset and Hugging Face `Trainer`.

```bash
uv lock
primejob dataset push data --disk pj-hf-sentiment --subdir data
primejob run train.py --gpu CPU --disk pj-hf-sentiment --data-mode stage --plain --yes
```

The script expects `PRIMEJOB_DATASET_PATH/data/train.json`. Use `--data-mode attach` for a single sequential run, or `--data-mode stage` when you want multiple jobs to reuse the same source disk without holding it for the whole training run.
