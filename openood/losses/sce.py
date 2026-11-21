
import torch
from torch.autograd.function import FunctionCtx


class SoftCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        logit: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert logit.size() == label.size(), "logit.size() != label.size()"
        dim = logit.dim()
        max_logit = logit.max(dim - 1, keepdim=True)[0]
        logit = logit - max_logit
        exp_logit = logit.exp()
        exp_sum = exp_logit.sum(dim - 1, keepdim=True)
        prob = exp_logit / exp_sum
        log_exp_sum = exp_sum.log()
        neg_log_prob = log_exp_sum - logit

        if weight is None:
            weighted_label = label
        else:
            if weight.size() != (logit.size(-1),):
                raise ValueError(
                    "since logit.size() = {}, "
                    "weight.size() should be ({},), but got {}".format(
                        logit.size(),
                        logit.size(-1),
                        weight.size(),
                    )
                )
            size = [1] * label.dim()
            size[-1] = label.size(-1)
            weighted_label = label * weight.view(size)
        ctx.save_for_backward(weighted_label, prob)
        out = (neg_log_prob * weighted_label).sum(dim - 1)
        # pdb.set_trace()
        return out

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        weighted_label, prob = ctx.saved_tensors
        old_size = weighted_label.size()
        # num_classes
        K = old_size[-1]
        # batch_size
        B = weighted_label.numel() // K

        grad_output = grad_output.view(B, 1)
        weighted_label = weighted_label.view(B, K)
        prob = prob.view(B, K)
        grad_input = grad_output * (prob * weighted_label.sum(1, True) - weighted_label)
        grad_input = grad_input.view(old_size)
        return grad_input, None, None


def soft_cross_entropy(
    logit: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if weight is not None and weight.requires_grad:
        raise RuntimeError("gradient for weight is not supported")
    losses = SoftCrossEntropyFunction.apply(logit, label, weight)
    reduction = {
        True: "mean",
        False: "none",
        None: reduction,
    }[reduce]
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError("invalid value for reduction: {}".format(reduction))


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight is None:
            weight = self.weight
        return soft_cross_entropy(logit, label, weight, self.reduce, self.reduction)
