import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fhk",
    version="0.0.1",
    author="Fabion Kauker",
    author_email="f.kauker@gmail.com",
    description="Run US wide analysis on broadband footprint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fhk/tabby_cat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "attrs==19.3.0",
        "beautifulsoup4==4.9.1",
        "certifi==2020.4.5.1",
        "cffi==1.14.0",
        "chardet==3.0.4",
        "click==7.1.2",
        "click-plugins==1.1.1",
        "cligj==0.5.0",
        "cryptography==2.9.2",
        "idna==2.9",
        "mkl-service==2.3.0",
        "munch==2.5.0",
        "networkx==2.4",
        "numpy==1.18.1",
        "pandas==1.0.3",
        "pcst-fast==1.0.7",
        "pybind11==2.5.0",
        "pycparser==2.20",
        "pyOpenSSL==19.1.0",
        "pyproj==2.6.1.post1",
        "PySocks==1.7.1",
        "python-dateutil==2.8.1",
        "pytz==2020.1",
        "requests==2.23.0",
        "Rtree==0.9.4",
        "Shapely==1.6.4.post1",
        "six==1.15.0",
        "soupsieve==2.0.1",
        "urllib3==1.25.8",
    ],
)
