"""Setup script for SimpleGNN package."""

from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simplegnn",
    version="0.1.0",
    author="SimpleGNN Contributors",
    description="A simple way to run predefined and new custom GNNs on benchmark datasets",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/fseiffarth/SimpleGNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here
    ],
)
