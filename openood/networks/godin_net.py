import torch
from torch import nn


def norm(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + 0.0001)
    return x


class CosineDeconf(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        w = norm(self.h.weight)

        ret = torch.matmul(x, w.T)
        return ret


class EuclideanDeconf(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # size: (batch, latent, 1)
        x = x.unsqueeze(2)

        # size: (1, latent, num_classes)
        h = self.h.weight.T.unsqueeze(0)
        ret = -((x - h).pow(2)).mean(1)
        return ret


class InnerDeconf(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity="relu")
        self.h.bias.data = torch.zeros(size=self.h.bias.size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(x)


class GodinNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        num_classes: int,
        similarity_measure: str = "cosine",
    ) -> None:
        super(GodinNet, self).__init__()

        h_dict = {
            "cosine": CosineDeconf,
            "inner": InnerDeconf,
            "euclid": EuclideanDeconf,
        }

        self.num_classes = num_classes

        self.backbone = backbone
        if hasattr(self.backbone, "fc"):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        self.h = h_dict[similarity_measure](feature_size, num_classes)

        self.g = nn.Sequential(
            nn.Linear(feature_size, 1), nn.BatchNorm1d(1), nn.Sigmoid()
        )

        self.softmax = nn.Softmax()

    def forward(
        self, x: torch.Tensor, inference: bool = False, score_func: str = "h"
    ) -> torch.Tensor:
        _, feature = self.backbone(x, return_feature=True)

        numerators = self.h(feature)

        denominators = self.g(feature)

        # calculate the logits results
        quotients = numerators / denominators

        # logits, numerators, and denominators
        if inference:
            if score_func == "h":
                return numerators
            elif score_func == "g":
                return denominators
            else:
                # maybe generate an error instead
                print("Invalid score function, using h instead")
                return numerators
        else:
            return quotients
