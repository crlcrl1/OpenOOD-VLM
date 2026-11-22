"""Encode both new prompt banks using the original per-label CLIP text pipeline."""

import torch
from capture import uniform_bank
from capture_features import CACHE

from openood.networks.clip_fixed_ood_prompt import (
    get_selected_ood_text_list,
    get_text_features_neg,
    load_clip_to_cpu,
)


@torch.no_grad()
def main() -> None:
    bank = uniform_bank()
    count = bank["selected"] - bank["id_classes"]
    words = get_selected_ood_text_list("cifar100", count)
    if [word for group in words for word in group] != bank["words"]:
        raise ValueError("WordNet candidate order changed relative to the baseline")
    model = load_clip_to_cpu("ViT-B/16").cuda().eval()
    out = CACHE / "multimodal"
    out.mkdir(exist_ok=True)
    for prompt in ("vanilla", "nice_vanilla"):
        chosen, rest = get_text_features_neg(
            model, "cifar100", prompt, True, count, uniform_negative_prompt=True
        )
        texts = torch.cat((chosen, rest), dim=1).cpu()
        if chosen.shape[1] != bank["selected"] or texts.shape != bank["texts"].shape:
            raise ValueError(f"Different number of labels for {prompt}")
        if not torch.isfinite(texts).all():
            raise ValueError(f"Nonfinite CLIP vector for {prompt}")
        path = out / f"{prompt}.pt"
        temp = path.with_suffix(".tmp")
        torch.save(texts, temp)
        temp.replace(path)
        print(f"Saved {prompt}: {texts.shape[1]} text vectors", flush=True)


if __name__ == "__main__":
    main()
