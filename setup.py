from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="taggle",
    version="0.1.0",
    # license="",
    description="A library that simplifies modeling and learning of image recognition in Pytorch",
    author="tattaka",
    url="https://github.com/tattaka/Taggle",
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
)
