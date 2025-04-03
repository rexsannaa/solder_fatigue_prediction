#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
setup.py - 銲錫接點疲勞壽命預測專案安裝檔
"""

from setuptools import setup, find_packages

setup(
    name="solder_fatigue_prediction",
    version="0.1.0",
    description="銲錫接點疲勞壽命預測混合PINN-LSTM模型",
    author="專案研發團隊",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/solder_fatigue_prediction",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "torch>=1.10.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "plotly>=5.3.0",
        "pathlib>=1.0.1",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "tensorboard>=2.7.0",
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    scripts=[
        "scripts/train.py",
        "scripts/predict.py",
        "scripts/evaluate.py",
    ],
    entry_points={
        "console_scripts": [
            "solder-train=scripts.train:main",
            "solder-predict=scripts.predict:main",
            "solder-evaluate=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)