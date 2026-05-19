# Unsloth SFT

Tiny Unsloth GPU smoke test using `unsloth/tinyllama-bnb-4bit`.

```bash
uv lock
primejob run train.py --gpu H200 --country FI --plain --yes
```

The example uses a four-row in-memory dataset and runs two SFT steps. It is meant to prove CUDA, Unsloth, TRL, upload, logging, outputs, and cleanup all work end-to-end.
