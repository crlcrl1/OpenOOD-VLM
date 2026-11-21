# model settings
model = {
    "type": "ImageClassifierWithReturnFeature",
    "backbone": {
        "type": "VisionTransformer",
        "arch": "b",
        "img_size": 384,
        "patch_size": 16,
        "drop_rate": 0.1,
        "init_cfg": [
            {
                "type": "Kaiming",
                "layer": "Conv2d",
                "mode": "fan_in",
                "nonlinearity": "linear",
            }
        ],
    },
    "neck": None,
    "head": {
        "type": "VisionTransformerClsHead",
        "num_classes": 1000,
        "in_channels": 768,
        "loss": {
            "type": "LabelSmoothLoss",
            "label_smooth_val": 0.1,
            "mode": "classy_vision",
        },
    },
}
