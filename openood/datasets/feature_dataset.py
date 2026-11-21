from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class FeatureSample(TypedDict):
    data: Tensor | np.ndarray
    label: int | np.integer | Tensor | np.ndarray


class FeatDataset(Dataset[FeatureSample]):
    def __init__(
        self, feat: Tensor | np.ndarray, labels: Tensor | np.ndarray | Sequence[int]
    ) -> None:
        self.data = feat
        self.labels = labels
        self.len = feat.shape[0]
        assert self.len == len(labels)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> FeatureSample:
        data = self.data[idx]
        label = self.labels[idx]
        return {"data": data, "label": label}
