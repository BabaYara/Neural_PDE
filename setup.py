from setuptools import setup, find_packages

setup(
    name="kfacpinn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["jax", "optax"],
    extras_require={
        "dev": ["pytest"],
    },
)
