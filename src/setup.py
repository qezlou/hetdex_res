from setuptools import setup, find_packages

setup(
    name="het_cov",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy",
        "h5py",
        "matplotlib"
    ],
)