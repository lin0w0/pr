"""Tongji ROI dataset returning raw ``[0,1]`` tensors and identity labels."""
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.preprocessing import load_raw_image
from utils.paths import resolve_path


@dataclass(frozen=True)
class Sample:
    path: str          # POSIX path relative to the project root
    label: int         # closed-set identity index in [0, num_identities)
    raw_id: int        # 1-based numeric filename prefix
    session: str
    index_in_id: int   # 0-based position within the identity's sorted images

    @property
    def abs_path(self) -> Path:
        return resolve_path(self.path)

    @property
    def filename(self) -> str:
        return Path(self.path).name


class TongjiDataset(Dataset):
    """Minimal dataset: returns ``(raw_image, label)``.

    ``raw_image`` is ``[1, H, W]`` float32 in ``[0, 1]``. Model preprocessing is
    applied later, on-device, so attacks can backpropagate through it.
    """

    def __init__(self, samples, input_size=128):
        self.samples = list(samples)
        self.input_size = int(input_size)
        if not self.samples:
            raise ValueError("empty sample list")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return load_raw_image(sample.path, self.input_size), sample.label

    def meta(self, index):
        return self.samples[index]

    @classmethod
    def from_samples(cls, samples, input_size=128):
        return cls(samples, input_size=input_size)

    @classmethod
    def from_csv(cls, csv_path, input_size=128):
        return cls(read_split_csv(csv_path), input_size=input_size)


def read_split_csv(csv_path):
    """Read a split CSV written by :func:`datasets.dataset_split.save_split`."""
    abs_path = resolve_path(csv_path)
    samples = []
    with open(abs_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label", "raw_id", "session", "index_in_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{abs_path} must contain columns {sorted(required)}")
        for row in reader:
            samples.append(Sample(
                path=row["path"],
                label=int(row["label"]),
                raw_id=int(row["raw_id"]),
                session=row["session"],
                index_in_id=int(row["index_in_id"]),
            ))
    return samples
