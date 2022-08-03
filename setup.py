#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages


setup(
    name='heatx',
    version='0.1.0',
    description='Model-agnostic explanation based on heat diffusion.',
    author='Rui Xia, Fan Yang',
    author_email='{guangyao.xr|fanyang.yf}@alibaba-inc.com',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'jax',
        'flax'
    ],
    packages=find_packages()
)
