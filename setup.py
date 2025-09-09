# File: setup.py
from setuptools import setup, find_packages

setup(
    name="hmm-variant-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A HMM-based A->G variant detector for NGS sequencing data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hmm-variant-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
)