from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


MODEL_NAME = "hf-internal-testing/tiny-random-distilbert"


def dataset_file() -> Path:
    root = os.environ.get("PRIMEJOB_DATASET_PATH")
    if not root:
        raise RuntimeError("PRIMEJOB_DATASET_PATH is missing. Run with --disk or upload/stage a dataset.")
    path = Path(root) / "data" / "train.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class JsonDataset:
    def __init__(self, path: Path, tokenizer) -> None:
        rows = json.loads(path.read_text(encoding="utf-8"))
        encoded = tokenizer([row["text"] for row in rows], padding=True, truncation=True, max_length=64)
        self.items = [
            {**{key: value[i] for key, value in encoded.items()}, "labels": int(rows[i]["label"])}
            for i in range(len(rows))
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return {key: torch.tensor(value) for key, value in self.items[index].items()}


def main() -> int:
    started = time.monotonic()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    train_dataset = JsonDataset(dataset_file(), tokenizer)

    output_dir = Path("outputs") / "hf_sentiment"
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=5,
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=11,
    )
    result = Trainer(model=model, args=args, train_dataset=train_dataset).train()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "model": MODEL_NAME,
        "rows": len(train_dataset),
        "train_loss": result.training_loss,
        "elapsed_s": round(time.monotonic() - started, 2),
        "dataset_path": str(dataset_file()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
