import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood._typing import Metrics
from openood.utils import Config, comm

from .lr_scheduler import cosine_annealing


class ARPLTrainer:
    def __init__(
        self, net: dict[str, nn.Module], train_loader: DataLoader, config: Config
    ) -> None:

        self.net = net["netF"]
        self.train_loader = train_loader
        self.config = config
        self.criterion = net["criterion"]

        params_list = [
            {"params": self.net.parameters()},
            {"params": self.criterion.parameters()},
        ]

        self.optimizer = torch.optim.SGD(
            params_list,
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    def train_epoch(self, epoch_idx: int) -> tuple[dict[str, nn.Module], Metrics]:
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # forward
            _, feat = self.net(data, return_feature=True)
            logits, loss = self.criterion(feat, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics: Metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = loss_avg

        return {"netF": self.net, "criterion": self.criterion}, metrics
