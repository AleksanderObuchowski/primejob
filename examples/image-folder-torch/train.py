from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def dataset_root() -> Path:
    root = os.environ.get("PRIMEJOB_DATASET_PATH")
    if not root:
        raise RuntimeError("PRIMEJOB_DATASET_PATH is missing. Run with --disk or upload/stage a dataset.")
    path = Path(root) / "data" / "image_folder"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_ppm(path: Path) -> torch.Tensor:
    parts = path.read_text(encoding="ascii").split()
    if parts[0] != "P3":
        raise ValueError(f"Expected P3 PPM image: {path}")
    width, height, max_value = int(parts[1]), int(parts[2]), int(parts[3])
    values = [int(value) / max_value for value in parts[4:]]
    return torch.tensor(values, dtype=torch.float32).view(height, width, 3).permute(2, 0, 1)


class ImageFolderDataset:
    def __init__(self, root: Path) -> None:
        self.classes = sorted(path.name for path in root.iterdir() if path.is_dir())
        self.samples: list[tuple[Path, int]] = []
        for index, class_name in enumerate(self.classes):
            self.samples.extend((path, index) for path in sorted((root / class_name).glob("*.ppm")))
        if not self.samples:
            raise RuntimeError(f"No .ppm files found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return read_ppm(path), torch.tensor(label, dtype=torch.long)


def main() -> int:
    started = time.monotonic()
    dataset = ImageFolderDataset(dataset_root())
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, len(dataset.classes)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(8):
        correct = 0
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
        losses.append(total / len(dataset))
        print(f"epoch={epoch + 1} loss={losses[-1]:.4f} accuracy={correct / len(dataset):.3f}", flush=True)

    output_dir = Path("outputs") / "image_folder"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dataset_root": str(dataset_root()),
        "classes": dataset.classes,
        "rows": len(dataset),
        "final_loss": losses[-1],
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), output_dir / "tiny_cnn.pt")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
