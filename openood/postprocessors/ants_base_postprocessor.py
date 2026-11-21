import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood._typing import LoaderDict, OODLoaders, ScoreArrays
from openood.utils import comm
from openood.utils.config import Config


class ANTSBasePostprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config

    def setup(
        self, net: nn.Module, id_loader_dict: LoaderDict, ood_loader_dict: OODLoaders
    ) -> None:
        pass

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(
        self, net: nn.Module, data_loader: DataLoader, progress: bool = True
    ) -> ScoreArrays:
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(
            data_loader, disable=not progress or not comm.is_main_process()
        ):
            data = batch["data"].cuda()
            label = batch["label"].cuda()
            path = batch["path"]
            pred, conf = self.postprocess(net, data, path)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
