from setuptools import setup

setup(
    name="louvain-python",
    version="0.0.1",
    url="http://github.com/shogo-ma/louvain-python",
    description="A implementation of louvain method on python",
    author="shogo-ma",
    packages=["louvain"],
    install_requires=["networkx"],
)
