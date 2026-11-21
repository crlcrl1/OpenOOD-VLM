import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openood-vlm",
    version="1.0",
    author="openood dev team",
    author_email="ybzhang815@gmail.com",
    description="This package provides a unified test platform for Out-of-Distribution detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YBZH/OpenOOD-VLM",
    packages=setuptools.find_packages(),
    python_requires=">=3.13",
    package_data={"openood": ["py.typed"]},
    install_requires=[
        "torch>=1.13.1",
        "torchvision>=0.13",
        "numpy",
        "Pillow",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pyyaml>=5.4.1",
        "opencv-python>=4.4.0.46",
        "imgaug>=0.4.0",
        "pandas",
        "diffdist>=0.1",
        "faiss-gpu>=1.7.2",
        "gdown>=4.7.1",
        "libmr>=0.1.9",
        "clip @ git+https://github.com/openai/CLIP.git",
        "ftfy",
        "regex",
        "open-clip-torch",
    ],
    extras_require={
        "ants": ["requests", "transformers>=4.45,<5"],
        "corpus": ["nltk", "requests"],
        "dev": ["pre-commit"],
        "openmmlab": ["mmpretrain[mim]>=1.2,<2"],
        "plot": ["matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
