"""Types shared by the training and evaluation interfaces."""

from collections.abc import Callable
from typing import Protocol, TypedDict

import numpy as np
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

type StateDict = dict[str, Tensor]
type Network = nn.Module | dict[str, nn.Module]
type LoaderDict = dict[str, DataLoader]
type ImageOutput = (
    Tensor | list[Tensor] | tuple[Tensor, ...] | np.ndarray | dict[str, np.ndarray]
)
type ImageTransform[Result = ImageOutput] = Callable[[Image.Image], Result]
type ScoreArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
type Metrics = dict[str, float | np.floating | Tensor]


class OODLoaders(TypedDict, total=False):
    """Validation uses a single loader. Evaluation splits group loaders by dataset."""

    val: DataLoader
    nearood: LoaderDict
    farood: LoaderDict
    csid: LoaderDict
    near: LoaderDict
    far: LoaderDict


class EvaluationLoaders(TypedDict):
    id: LoaderDict
    csid: LoaderDict
    ood: OODLoaders


class GanMetrics(TypedDict):
    epoch_idx: int
    G_losses: list[float]
    D_losses: list[float]


class Trainer[EpochResult](Protocol):
    def train_epoch(self, epoch_idx: int) -> EpochResult: ...


class Recorder[Model, EpochMetrics](Protocol):
    def report(self, train_metrics: EpochMetrics, val_metrics: Metrics, /) -> None: ...
    def save_model(self, net: Model, metrics: Metrics, /) -> None: ...
    def summary(self) -> None: ...


class Pipeline(Protocol):
    def run(self) -> None: ...
