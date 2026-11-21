from openood._typing import ImageTransform
from openood.utils import Config

from .augmix_preprocessor import AugMixPreprocessor
from .base_preprocessor import BasePreprocessor
from .cider_preprocessor import CiderPreprocessor
from .csi_preprocessor import CSIPreprocessor
from .cutout_preprocessor import CutoutPreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .pixmix_preprocessor import PixMixPreprocessor
from .randaugment_preprocessor import RandAugmentPreprocessor
from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(config: Config, split: str) -> ImageTransform:
    train_preprocessors = {
        "base": BasePreprocessor,
        "draem": DRAEMPreprocessor,
        "cutpaste": CutPastePreprocessor,
        "augmix": AugMixPreprocessor,
        "pixmix": PixMixPreprocessor,
        "randaugment": RandAugmentPreprocessor,
        "cutout": CutoutPreprocessor,
        "csi": CSIPreprocessor,
        "cider": CiderPreprocessor,
    }
    test_preprocessors = {
        "base": TestStandardPreProcessor,
        "draem": DRAEMPreprocessor,
        "cutpaste": CutPastePreprocessor,
    }

    if split == "train":
        return train_preprocessors[config.preprocessor.name](config)
    else:
        try:
            return test_preprocessors[config.preprocessor.name](config)
        except KeyError:
            return test_preprocessors["base"](config)
