from setuptools import setup, find_packages

setup(
    name="soara-peft",
    version="0.1.0",
    description="SOARA: Subspace Orthogonal Adaptation via Rotational Alignment — Parameter-Efficient Fine-Tuning via rotational alignment in SVD subspaces",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chandan22140/soara-peft",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "train": [
            "wandb",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "pandas",
            "torchvision",
            "timm",
            "Pillow",
        ],
        "quantize": [
            "bitsandbytes>=0.41.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
