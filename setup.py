from setuptools import setup, find_packages
import pathlib

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    name="x86cpu",  # Required
    version="1.0.0",  # Required
    description="A sample Python project",  # Optional
    url="https://github.com/hopkiw/x86cpu",  # Optional
    author="Liam Hopkins",
    author_email="we.hopkins@gmail.com",
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.7, <4",
)
