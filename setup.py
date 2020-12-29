# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name="adcp-nav",
    version="untracked",
    description="Package to calculate vehicle positioning and current "
    "profiles of undersea vehicles equipped with acoustic "
    "doppler current profiler using Kalman Filter.",
    author="Jonathan Jonker, Jake Stevens-Haas, Sarah Webster, "
    "Aleksandr Aravkin",
    author_email="jmsh@uw.com",
    license="None",
    packages=[
        "adcp",
        "adcp/tests",
    ],  # Packages & subpackages for the directory
    # containintby setup.py
    py_modules=["dataprep", "matbuilder"],  # This package's modules
    install_requires=[
        "numpy(>=1.16)",
        "pandas(>=0.23)",
        "scipy(>=1.2)",
        "matplotlib(>=3.0)",
        "h5py(>=2.0)",
    ],
    zip_safe=False,
    python_requires="~=3.6",
)
