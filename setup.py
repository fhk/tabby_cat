import setuptools
from setuptools import Extension
import sys
import os

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'graph_assembler_cpp',
        ['src/graph_assembler.cpp', 'src/bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            '/usr/include/gdal',
            '/usr/include/osmium',
        ],
        libraries=['gdal', 'z', 'expat', 'bz2', 'spatialindex'],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tabby_cat",
    version="0.0.1",
    author="Fabion Kauker",
    author_email="f.kauker@gmail.com",
    description="Run US wide analysis on broadband footprint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fhk/tabby_cat",
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    setup_requires=['pybind11'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "attrs",
        "beautifulsoup4",
        "certifi",
        "cffi",
        "chardet",
        "click",
        "click-plugins",
        "cligj",
        "cryptography",
        "h3",
        "geopandas",
        "idna",
        "munch",
        "networkx",
        "numpy",
        "pandas",
        "pcst-fast",
        "pybind11",
        "pycparser",
        "pyOpenSSL",
        "pyproj",
        "PySocks",
        "pytest",
        "python-dateutil",
        "pytz",
        "requests",
        "Rtree",
        "scipy",
        "Shapely",
        "six",
        "soupsieve",
        "urllib3",
        "pyarrow",
        "fastparquet"
    ],
)
