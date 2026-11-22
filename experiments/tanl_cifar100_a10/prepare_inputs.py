"""Prepare exactly the public inputs used for TANL Table A10(b)."""

import csv
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

EXPERIMENT = Path(__file__).resolve().parent
DATA = EXPERIMENT.parents[1] / "data"
DOWNLOADS = DATA / "tanl_a10_downloads"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_inputs(sources: Path = EXPERIMENT / "sources.tsv") -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    with sources.open() as manifest:
        for name, expected, url in csv.reader(manifest, delimiter="\t"):
            target = DOWNLOADS / name
            if target.exists():
                if digest(target) != expected:
                    raise ValueError(f"Unexpected SHA-256 for {target}")
                continue
            print(f"Downloading {name}", flush=True)
            with TemporaryDirectory(dir=DOWNLOADS) as folder:
                source = Path(folder) / name
                with urlopen(url, timeout=90) as response, source.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if digest(source) != expected:
                    raise ValueError(f"Download did not match SHA-256 for {name}")
                source.replace(target)


def prepare_lists() -> None:
    target = DATA / "benchmark_imglist" / "cifar100"
    target.mkdir(parents=True, exist_ok=True)
    labels = {}
    for name in ("test_cifar100.txt", "val_cifar100.txt"):
        for row in (DOWNLOADS / name).read_text().splitlines():
            image, label = row.split()
            labels[image] = label
    order = [
        row.split()[0]
        for row in (DOWNLOADS / "full_cifar100_crossref.txt").read_text().splitlines()
    ]
    if len(order) != 10000 or len(labels) != 10000 or set(order) != labels.keys():
        raise ValueError("CIFAR-100 full split did not match the published lists")
    (target / "test_cifar100_full.txt").write_text(
        "".join(f"{image} {labels[image]}\n" for image in order)
    )
    shutil.copyfile(DOWNLOADS / "test_cifar10.txt", target / "test_cifar10.txt")


def main() -> None:
    fetch_inputs()
    for dataset in ("cifar10", "cifar100"):
        with ZipFile(DOWNLOADS / f"{dataset}.zip") as archive:
            for image in archive.namelist():
                if image.startswith("test/") and image.endswith(".png"):
                    archive.extract(image, DATA / "images_classic" / dataset)
    words = DATA / "txtfiles"
    words.mkdir(exist_ok=True)
    with ZipFile(DOWNLOADS / "NegLabel-3253db6.zip") as archive:
        for entry in archive.namelist():
            path = Path(entry)
            if path.parent.name == "txtfiles" and path.name.startswith(("adj", "noun")):
                (words / path.name).write_bytes(archive.read(entry))
    prepare_lists()
    checkpoint = DOWNLOADS / "ViT-B-16.pt"
    clip_cache = Path.home() / ".cache" / "clip" / checkpoint.name
    if not clip_cache.exists():
        clip_cache.parent.mkdir(parents=True, exist_ok=True)
        clip_cache.symlink_to(checkpoint)
    elif digest(clip_cache) != digest(checkpoint):
        raise ValueError(f"Unexpected CLIP weights at {clip_cache}")


if __name__ == "__main__":
    main()
