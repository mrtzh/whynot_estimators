"""Build whynot_estimators package."""
from setuptools import setup, find_packages

setup(
    name="whynot_estimators",
    version="0.11.0",
    author="John Miller",
    author_email="miller_john@berkeley.edu",
    description="Companion package to whynot: A collection of causal estimators.",
    packages=find_packages(),
    install_requires=["cython", "numpy", "pandas", "sklearn", "whynot", "tzlocal",],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
    ],
    python_requires=">=3.6",
)
