import setuptools
from setuptools import setup

setup(
    name="msc-project",
    version="0.0.1",
    description="Source code for expirements made in our Master's Thesis at Aarhus University",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="multi-agent reinforcement learning pytorch partial observability",
    python_requires=">=3.6",
)
