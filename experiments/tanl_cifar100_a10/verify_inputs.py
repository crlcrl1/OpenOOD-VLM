"""Cross-check the frozen inputs against Toronto's CIFAR test splits and OpenAI CLIP."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import clip
import numpy as np
import pyarrow.parquet as pq
import torch
from capture_features import CACHE, diagnostic_config
from PIL import Image
from run import EXPERIMENT, set_seed
from torch.utils.data import DataLoader

from openood.networks.clip import clip as repo_clip
from openood.networks.clip_fixed_ood_prompt import get_class_names
from openood.preprocessors.utils import get_preprocessor

SPLITS = {
    "cifar10": (
        "0b2714987fa478483af9968de7c934580d0bb9a2/plain_text",
        "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
    ),
    "cifar100": (
        "aadb3af77e9048adbea6b47c21a81e47dd092ae5/cifar100",
        "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68",
    ),
}


def pixel_audit(dataset):
    revision_path, expected = SPLITS[dataset]
    url = f"https://huggingface.co/datasets/uoft-cs/{dataset}/resolve/{revision_path}/test-00000-of-00001.parquet"
    path = CACHE / f"{dataset}-test.parquet"
    if not path.exists():
        urlretrieve(url, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected reference split at {path}")
    classes = get_class_names(dataset)
    key = "fine_label" if dataset == "cifar100" else "label"
    reference = Counter()
    for row in pq.read_table(path, columns=["img", key]).to_pylist():
        with Image.open(io.BytesIO(row["img"]["bytes"])) as im:
            reference[
                hashlib.sha256(im.convert("RGB").tobytes()).digest(), row[key]
            ] += 1
    local = Counter()
    labels = []
    paths = []
    filename = "test_cifar100_full.txt" if dataset == "cifar100" else "test_cifar10.txt"
    manifest = Path("data/benchmark_imglist/cifar100") / filename
    for entry in manifest.read_text().splitlines():
        relative, label = entry.split()
        image = Path("data/images_classic") / relative
        class_index = classes.index(image.parent.name)
        target = class_index if dataset == "cifar100" else -1
        if int(label) != target:
            raise ValueError(f"Label mismatch at {relative}")
        with Image.open(image) as im:
            local[
                hashlib.sha256(im.convert("RGB").tobytes()).digest(), class_index
            ] += 1
        paths.append(image)
        labels.append(target)
    matches = (local & reference).total()
    if not matches == local.total() == reference.total() == 10000:
        raise ValueError(
            f"Only {matches} RGB images and class IDs matched on {dataset}"
        )
    return paths, labels, {"matches": matches, "sha256": expected, "url": url}


@torch.no_grad()
def main():
    paths, labels, pixels = [], [], {}
    for dataset in SPLITS:  # OpenOOD concatenates OOD first, then ID.
        files, targets, pixels[dataset] = pixel_audit(dataset)
        paths.extend(files)
        labels.extend(targets)
    config = diagnostic_config()
    size = config.dataset.test.batch_size
    stream = torch.load(CACHE / "stream0.pt")
    set_seed(0)
    order = torch.cat(
        list(DataLoader(range(len(paths)), shuffle=True, batch_size=size))
    )
    if not torch.equal(torch.tensor(labels)[order], stream["labels"]):
        raise ValueError("Independent replay did not align with all seed-0 labels")
    torch.backends.cudnn.benchmark = True
    checkpoint = Path("data/tanl_a10_downloads/ViT-B-16.pt")
    model, official_preprocess = clip.load(str(checkpoint), device="cuda", jit=False)
    repo_preprocess = get_preprocessor(config, "test")
    images = []
    for index in order[:size]:
        with Image.open(paths[index]) as im:
            rgb = im.convert("RGB")
            standard = official_preprocess(rgb)
            if not torch.equal(standard, repo_preprocess(rgb)):
                raise ValueError("The preprocessing tensors differ")
            images.append(standard)
    reference_images = model.encode_image(torch.stack(images).cuda())
    reference_images /= reference_images.norm(dim=-1, keepdim=True)
    bank = torch.load(CACHE / "bank.pt")
    vocabulary = torch.load(
        "data/txtfiles_output/wordnet_cifar100_cossim_dedup.pth", map_location="cpu"
    )
    adjectives = set(vocabulary["text_list_adj"])
    indices = np.linspace(0, len(bank["words"]) - 1, 128, dtype=int).tolist()
    prompts = [f"The nice {word}." for word in get_class_names("cifar100")]
    for i in indices:
        word = bank["words"][i]
        prompts.append(
            f"This is a {word} photo." if word in adjectives else f"The nice {word}."
        )
    tokens = clip.tokenize(prompts)
    if not torch.equal(tokens, repo_clip.tokenize(prompts)):
        raise ValueError("The tokenizers differ")
    # The release encodes a single prompt at a time and normalizes a second time
    # after averaging the one-element template list.
    vectors = torch.cat([model.encode_text(t[None].cuda()) for t in tokens])
    vectors /= vectors.norm(dim=-1, keepdim=True)
    vectors /= vectors.norm(dim=-1, keepdim=True)
    compare = [*range(bank["id_classes"]), *(i + bank["id_classes"] for i in indices)]
    text_difference = (vectors.cpu() - bank["texts"][:, compare].T).float().abs()
    image_difference = (reference_images.cpu() - stream["images"][:size]).float().abs()
    reference_logits = (
        model.logit_scale.exp() * reference_images @ vectors[: bank["id_classes"]].T
    )
    with np.load(EXPERIMENT / "scores_seed0.npz") as result:
        matches = int(
            np.sum(reference_logits.argmax(1).cpu().numpy() == result["pred"][:size])
        )
    report = {
        "reference_splits": pixels,
        "images_preprocessed_identically": size,
        "max_image_feature_difference": image_difference.max().item(),
        "prompts_tokenized_identically": len(prompts),
        "max_text_feature_difference": text_difference.max().item(),
        "matching_class_predictions": matches,
        "temperature": 1 / model.logit_scale.exp().item(),
    }
    path = EXPERIMENT / "diagnostics" / "input_audit.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
