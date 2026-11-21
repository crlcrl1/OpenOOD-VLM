from functools import partial
from typing import Literal, overload

from torch import Tensor, nn
from torchvision.models.regnet import BlockParams, RegNet


class RegNet_Y_16GF(RegNet):
    def __init__(self) -> None:
        block_params = BlockParams.from_init_params(
            depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25
        )
        norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
        super(RegNet_Y_16GF, self).__init__(
            block_params=block_params, norm_layer=norm_layer
        )

    @overload
    def forward(self, x: Tensor, return_feature: Literal[False] = False) -> Tensor: ...

    @overload
    def forward(
        self, x: Tensor, return_feature: Literal[True]
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, x: Tensor, return_feature: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        feas = x.flatten(start_dim=1)
        logits = self.fc(feas)

        if return_feature:
            return logits, feas
        else:
            return logits
