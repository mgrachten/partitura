#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "partitura"
DESCRIPTION = "A package for handling symbolic musical information"
KEYWORDS = "music notation musicxml midi"
URL = "https://github.com/CPJKU/partitura"
EMAIL = "partitura-users@googlegroups.com"
AUTHOR = "Maarten Grachten, Carlos Cancino-Chacón, Silvan Peter, Emmanouil Karystinaios, Francesco Foscarin, Thassilo Gadermaier, Patricia Hu"
REQUIRES_PYTHON = ">=3.7"
VERSION = "1.6.0"

# What packages are required for this module to be executed?
REQUIRED = ["numpy", "scipy", "lxml", "lark-parser", "xmlschema", "mido"]

# What packages are optional?
EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={
        "partitura": [
            "assets/musicxml.xsd",
            "assets/score_example.mid",
            "assets/score_example.musicxml",
            "assets/score_example.krn",
            "assets/score_example.mei",
        ]
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
