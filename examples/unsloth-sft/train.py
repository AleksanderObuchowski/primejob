from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import unsloth
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


_ = unsloth
MODEL_NAME = os.environ.get("UNSLOTH_MODEL", "unsloth/tinyllama-bnb-4bit")


def main() -> int:
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU pod.")

    max_seq_length = 256
    rows = [
        {"text": "### Instruction: Summarize the incident.\n### Response: The payment service recovered after cache rollback."},
        {"text": "### Instruction: Classify sentiment: users praised the faster release.\n### Response: positive"},
        {"text": "### Instruction: Classify sentiment: the dashboard crashed repeatedly.\n### Response: negative"},
        {"text": "### Instruction: Extract action item.\n### Response: Add a regression test for checkout retries."},
    ]
    dataset = Dataset.from_list(rows)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
    )

    output_dir = Path("outputs") / "unsloth_sft"
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            dataset_num_proc=1,
            max_length=max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=3407,
        ),
    )
    result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": MODEL_NAME,
        "rows": len(rows),
        "train_loss": result.training_loss,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
