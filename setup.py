#!/usr/bin/env python

# This file is copied and modified from the Koalas project:
#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function

import sys
from io import open
from setuptools import setup
from os import path

DESCRIPTION = "PyDDS: data-driven programing"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

try:
    exec(open("dds/_version.py").read())
except IOError:
    print(
        "Failed to load Koalas version file for packaging. You must be in dds root dir.",
        file=sys.stderr,
    )
    sys.exit(-1)
VERSION = version  # noqa

setup(
    name="dds_py",
    version=VERSION,
    packages=["dds", "dds.codecs"],
    package_data={"dds": ["py.typed"]},
    extras_require={
        "pandas": ["pandas>=0.23.1", "pyarrow>=0.10"],
        "spark": ["pyspark>=2.4.0"],
    },
    python_requires=">=3.6,<3.10",
    install_requires=["asttokens"],
    author="Tim Hunter",
    author_email="tjhunter+dds@cs.stanford.edu",
    license="AGPL",
    url="https://github.com/tjhunter/dds_py",
    project_urls={
        "Bug Tracker": "https://github.com/tjhunter/dds_py/issues",
        "Documentation": "https://tjhunter.github.io/dds_py/",
        "Source Code": "https://github.com/tjhunter/dds_py",
    },
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Typing :: Typed",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
