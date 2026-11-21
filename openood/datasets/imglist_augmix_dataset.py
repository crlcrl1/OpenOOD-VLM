import ast
import io
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL import Image, ImageFile

from openood._typing import ImageTransform

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode: str = "RGB") -> None:
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert(self.mode)


class ImglistAugMixDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        imglist_pth: str,
        data_dir: str,
        num_classes: int,
        preprocessor: ImageTransform,
        data_aux_preprocessor: Callable[[Image.Image], torch.Tensor],
        maxlen: int | None = None,
        dummy_read: bool = False,
        dummy_size: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super(ImglistAugMixDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError("if dummy_read is True, should provide dummy_size")

    def __len__(self) -> int:
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index: int) -> dict[str, Any]:
        line = self.imglist[index].strip("\n")
        tokens = line.split(" ", 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != "" and image_name.startswith("/"):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample: dict[str, Any] = dict()
        sample["image_name"] = image_name
        kwargs = {"name": self.name, "path": path, "tokens": tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if not self.dummy_read:
                with open(path, "rb") as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample["data"] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert("RGB")
                orig, aug1, aug2 = self.transform_image(image)
                sample["data"] = orig
                sample["data_aug1"] = aug1
                sample["data_aug2"] = aug2
                sample["data_aux"] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample["label"] = 0
            except AttributeError:
                sample["label"] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample["label"] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample["label"]] = 1
            sample["soft_label"] = soft_label

        except Exception as e:
            logging.error(f"[{path}] broken")
            raise e
        return sample
