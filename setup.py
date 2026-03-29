"""Setup script for CausalFair."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="causalfair",
    version="0.1.0",
    author="Duraimurugan Rajamanickam",
    author_email="drajamanicka@ualr.edu",
    description="Causal Mediation Analysis for Algorithmic Fairness in Credit Decisions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdmurugan/causalfair",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.4",
        "scikit-learn>=1.0",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.12",
        ],
    },
    keywords=[
        "causal inference",
        "mediation analysis",
        "algorithmic fairness",
        "credit scoring",
        "HMDA",
        "disparate impact",
    ],
)
