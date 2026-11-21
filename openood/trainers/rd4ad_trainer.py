
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood._typing import Metrics
from openood.losses.rd4ad_loss import loss_function
from openood.utils import Config


class Rd4adTrainer:
    def __init__(
        self, net: dict[str, torch.nn.Module], train_loader: DataLoader, config: Config
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.encoder = net["encoder"]
        self.bn = net["bn"]
        self.decoder = net["decoder"]
        if config.optimizer.name == "adam":
            self.optimizer = torch.optim.Adam(
                list(self.decoder.parameters()) + list(self.bn.parameters()),
                lr=config.optimizer.lr,
                betas=config.optimizer.betas,
            )

    def train_epoch(self, epoch_idx: int) -> tuple[dict[str, torch.nn.Module], Metrics]:
        self.encoder.eval()
        self.bn.train()
        self.decoder.train()
        train_dataiter = iter(self.train_loader)
        epoch_loss = 0
        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d} ".format(epoch_idx),
            position=0,
            leave=True,
        ):
            batch = next(train_dataiter)
            img = batch["data"].cuda()
            feature_list = self.encoder.forward(img, return_feature_list=True)[1]
            inputs = feature_list[1:4]
            outputs = self.decoder(self.bn(inputs))
            loss = loss_function(inputs, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        metrics: Metrics = {}
        net = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = epoch_loss
        net["encoder"] = self.encoder
        net["bn"] = self.bn
        net["decoder"] = self.decoder
        return net, metrics
