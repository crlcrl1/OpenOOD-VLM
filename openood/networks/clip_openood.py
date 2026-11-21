from collections.abc import Sequence

import clip
import torch
from clip.model import CLIP as ClipModel
from torch import nn
from tqdm import tqdm


# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
def zeroshot_classifier(
    model: ClipModel, classnames: Sequence[str], templates: Sequence[str]
) -> torch.Tensor:
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


class CLIPZeroshot(nn.Module):
    def __init__(
        self,
        classnames: Sequence[str],
        templates: Sequence[str],
        backbone: str = "ViT-B/16",
    ) -> None:
        super().__init__()
        assert backbone in clip.available_models()
        self.model, self.preprocess = clip.load(backbone, device="cuda")
        self.logit_scale = self.model.logit_scale.data
        self.zeroshot_weights = zeroshot_classifier(self.model, classnames, templates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.zeroshot_weights  # scaling ??
        return logits
